#!/usr/bin/env python3
"""Evaluate a full SurgVLP checkpoint on a Kvasir-style folder dataset."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from mmengine.config import Config
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# Avoid HF tokenizer fork warnings when DataLoader uses worker processes.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Ensure local package import works even when launched from outside repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

import surgvlp  # noqa: E402


class KvasirFolderDataset(Dataset):
    """Folder-per-class Kvasir dataset with recursive image discovery."""

    IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

    def __init__(self, root: str, class_order: List[str], transform=None):
        self.root = root
        self.transform = transform
        self.class_order = class_order
        self.class_to_idx = {c: i for i, c in enumerate(self.class_order)}

        self.samples: List[str] = []
        self.labels: List[int] = []
        self.missing_classes: List[str] = []

        for class_name in self.class_order:
            class_dir = self._resolve_class_dir(root, class_name)
            if class_dir is None:
                self.missing_classes.append(class_name)
                continue
            for dirpath, _, filenames in os.walk(class_dir):
                for fname in sorted(filenames):
                    if fname.lower().endswith(self.IMAGE_EXTS):
                        self.samples.append(os.path.join(dirpath, fname))
                        self.labels.append(self.class_to_idx[class_name])

        if not self.samples:
            raise RuntimeError(f"No images found under {self.root}")

    @staticmethod
    def _resolve_class_dir(root: str, canonical: str) -> Optional[str]:
        # Canonical prompts use hyphens, but many roots use underscores.
        candidates = [canonical, canonical.replace("-", "_"), canonical.replace("_", "-")]
        for cand in candidates:
            p = os.path.join(root, cand)
            if os.path.isdir(p):
                return p
        return None

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        path = self.samples[index]
        label = self.labels[index]
        with Image.open(path) as img:
            img = img.convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return {"video": img, "label": label}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate full SurgVLP checkpoint on Kvasir")
    parser.add_argument("--checkpoint", type=str, required=True, help="Full model checkpoint path (e.g., SurgVLP_d1.pth)")
    parser.add_argument("--config", type=str, default="/DATA3/SurgVLP-main/tests/config_surgvlp.py")
    parser.add_argument("--kvasir-root", type=str, required=True, help="Root of Kvasir subset to evaluate")
    parser.add_argument("--kvasir-prompts", type=str, default="/DATA3/SurgVLP-main/tests/class_prompt_kvasir.txt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--logit-scale", type=float, default=20.0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--output-json", type=str, default="")
    return parser.parse_args()


def resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(choice)


def default_eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def read_prompts(path: str) -> List[str]:
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            t = line.strip()
            if t and not t.startswith("#"):
                texts.append(t)
    if not texts:
        raise RuntimeError(f"No prompts found in {path}")
    return texts


def confusion_and_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    support = cm.sum(axis=1)
    tp = np.diag(cm).astype(np.float64)
    pred_pos = cm.sum(axis=0).astype(np.float64)
    actual_pos = support.astype(np.float64)

    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall = np.divide(tp, actual_pos, out=np.zeros_like(tp), where=actual_pos > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)

    class_acc = np.divide(tp, actual_pos, out=np.zeros_like(tp), where=actual_pos > 0)
    macro_f1 = float(f1.mean()) if num_classes > 0 else 0.0
    return class_acc.tolist(), f1.tolist(), support.tolist(), macro_f1


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    tokens: torch.Tensor,
    device: torch.device,
    logit_scale: float,
    label_smoothing: float,
):
    model.eval()
    txt = model(None, tokens, mode="text")["text_emb"]
    txt = F.normalize(txt, dim=-1)

    loss_sum = 0.0
    total = 0
    y_true = []
    y_pred = []

    for batch in loader:
        images = batch["video"].to(device, non_blocking=True)
        labels = batch["label"].long().to(device, non_blocking=True)
        img = model(images, None, mode="video")["img_emb"]
        img = F.normalize(img, dim=-1)
        logits = logit_scale * (img @ txt.T)
        loss = F.cross_entropy(logits, labels, label_smoothing=label_smoothing)

        preds = logits.argmax(dim=1)
        bsz = labels.numel()
        loss_sum += loss.item() * bsz
        total += bsz
        y_true.append(labels.cpu().numpy())
        y_pred.append(preds.cpu().numpy())

    if total == 0:
        return {
            "num_samples": 0,
            "loss": 0.0,
            "acc": 0.0,
            "macro_f1": 0.0,
            "class_f1": [],
            "class_acc": [],
            "class_support": [],
        }

    y_true_arr = np.concatenate(y_true, axis=0)
    y_pred_arr = np.concatenate(y_pred, axis=0)
    num_classes = int(tokens["input_ids"].shape[0]) if isinstance(tokens, dict) else int(tokens.shape[0])

    class_acc, class_f1, class_support, macro_f1 = confusion_and_metrics(y_true_arr, y_pred_arr, num_classes)
    acc = float((y_true_arr == y_pred_arr).mean())

    return {
        "num_samples": int(total),
        "loss": float(loss_sum / total),
        "acc": acc,
        "macro_f1": macro_f1,
        "class_f1": class_f1,
        "class_acc": class_acc,
        "class_support": class_support,
    }


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    cfg = Config.fromfile(args.config)["config"]
    model, _ = surgvlp.load(cfg.model_config, device=device, pretrain=args.checkpoint)
    model = model.to(device)

    kvasir_class_order = [
        "dyed-lifted-polyps",
        "dyed-resection-margins",
        "esophagitis",
        "normal-cecum",
        "normal-pylorus",
        "normal-z-line",
        "polyps",
        "ulcerative-colitis",
    ]
    ds = KvasirFolderDataset(
        root=args.kvasir_root,
        class_order=kvasir_class_order,
        transform=default_eval_transform(),
    )
    if ds.missing_classes:
        print(
            f"[Warn] missing_class_dirs={len(ds.missing_classes)} "
            f"({', '.join(ds.missing_classes)})"
        )
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    prompts = read_prompts(args.kvasir_prompts)
    tokens = surgvlp.tokenize(prompts, device=device)

    print(f"[Info] checkpoint={args.checkpoint}")
    print(f"[Info] config={args.config}")
    print(f"[Info] kvasir_root={args.kvasir_root}")
    print(f"[Info] logit_scale={args.logit_scale}")
    print(f"[Info] num_samples={len(ds)}")

    metrics = evaluate(
        model,
        loader,
        tokens,
        device=device,
        logit_scale=float(args.logit_scale),
        label_smoothing=float(args.label_smoothing),
    )
    print(
        f"[Result] loss={metrics['loss']:.4f} acc={metrics['acc']:.4f} "
        f"macro_f1={metrics['macro_f1']:.4f}"
    )
    print("[Classwise]")
    for i, cls_name in enumerate(kvasir_class_order):
        support = metrics["class_support"][i]
        acc = metrics["class_acc"][i]
        f1 = metrics["class_f1"][i]
        print(f"{i:02d} {cls_name}: support={support} acc={acc:.4f} f1={f1:.4f}")

    out = args.output_json
    if not out:
        subset_name = os.path.basename(os.path.normpath(args.kvasir_root))
        ckpt_stem = Path(args.checkpoint).stem
        out = os.path.join(os.path.dirname(args.checkpoint), f"{subset_name}_eval_{ckpt_stem}.json")

    payload = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "kvasir_root": args.kvasir_root,
        "kvasir_prompts": args.kvasir_prompts,
        "logit_scale": float(args.logit_scale),
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[Info] wrote {out}")


if __name__ == "__main__":
    main()
