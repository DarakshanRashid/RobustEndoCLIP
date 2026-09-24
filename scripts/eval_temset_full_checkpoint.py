#!/usr/bin/env python3
"""Evaluate a full SurgVLP checkpoint on a TEMSET-style root."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from mmengine.config import Config
from torch.utils.data import DataLoader

# Avoid HF tokenizer fork warnings when DataLoader uses worker processes.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

# Ensure local package import works even when launched from outside repo root.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(ROOT))

import surgvlp  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate full SurgVLP checkpoint on TEMSET")
    parser.add_argument("--checkpoint", type=str, required=True, help="Full model checkpoint path")
    parser.add_argument("--config", type=str, default="/DATA3/SurgVLP-main/tests/config_surgvlp.py")
    parser.add_argument("--ann-file", type=str, default="/DATA3/tc-clip-main/datasets_splits/temset_splits/val.txt")
    parser.add_argument("--video-root", type=str, required=True, help="TEMSET frame root to evaluate")
    parser.add_argument("--class-prompt", type=str, default="/DATA3/SurgVLP-main/tests/class_prompt_temset.txt")
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


def eval_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((360, 640)),
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
    tokens,
    device: torch.device,
    logit_scale: float,
    label_smoothing: float,
) -> Dict[str, object]:
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

    ds_cfg = dict(
        type="Recognition_temset",
        ann_file=args.ann_file,
        video_root=args.video_root,
        frame_policy="center",
        transforms=eval_transform(),
    )
    ds = surgvlp.load_dataset(ds_cfg)
    loader = DataLoader(
        ds,
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    prompts = read_prompts(args.class_prompt)
    tokens = surgvlp.tokenize(prompts, device=device)

    print(f"[Info] checkpoint={args.checkpoint}")
    print(f"[Info] ann_file={args.ann_file}")
    print(f"[Info] video_root={args.video_root}")
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

    out = args.output_json
    if not out:
        subset_name = os.path.basename(os.path.normpath(args.video_root))
        ckpt_stem = Path(args.checkpoint).stem
        out = os.path.join(os.path.dirname(args.checkpoint), f"{subset_name}_eval_{ckpt_stem}.json")

    payload = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "ann_file": args.ann_file,
        "video_root": args.video_root,
        "class_prompt": args.class_prompt,
        "logit_scale": float(args.logit_scale),
        "metrics": metrics,
    }
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[Info] wrote {out}")


if __name__ == "__main__":
    main()
