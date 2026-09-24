#!/usr/bin/env python3
"""Evaluate a full SurgVLP checkpoint on a CholecT50-style root."""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

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
    parser = argparse.ArgumentParser(description="Evaluate full SurgVLP checkpoint on CholecT50")
    parser.add_argument("--checkpoint", type=str, required=True, help="Full model checkpoint path")
    parser.add_argument("--config", type=str, default="/DATA3/SurgVLP-main/tests/config_surgvlp.py")
    parser.add_argument("--video-root", type=str, required=True, help="Root folder containing VID* dirs")
    parser.add_argument("--csv-root", type=str, default="/DATA3/SurgVLP-main/tests/cholect50/csvs")
    parser.add_argument("--class-prompt", type=str, default="/DATA3/SurgVLP-main/tests/class_prompt.txt")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--logit-scale", type=float, default=100.0)
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


def available_vids(video_root: str, csv_root: str) -> List[str]:
    vids = []
    for name in sorted(os.listdir(video_root)):
        if not name.startswith("VID"):
            continue
        if os.path.isdir(os.path.join(video_root, name)):
            csv_path = os.path.join(csv_root, f"video_{name}.csv")
            if os.path.exists(csv_path):
                vids.append(name)
    if not vids:
        raise RuntimeError(f"No matching VID* folders with CSV found under {video_root}")
    return vids


def video_metrics(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> Tuple[float, float]:
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    tp = np.diag(cm).astype(np.float64)
    support = cm.sum(axis=1).astype(np.float64)
    pred_pos = cm.sum(axis=0).astype(np.float64)
    precision = np.divide(tp, pred_pos, out=np.zeros_like(tp), where=pred_pos > 0)
    recall = np.divide(tp, support, out=np.zeros_like(tp), where=support > 0)
    f1 = np.divide(2 * precision * recall, precision + recall, out=np.zeros_like(tp), where=(precision + recall) > 0)
    acc = float((y_true == y_pred).mean()) if y_true.size > 0 else 0.0
    macro_f1 = float(f1.mean()) if num_classes > 0 else 0.0
    return acc, macro_f1


@torch.no_grad()
def evaluate_video(model, loader, text_features, device: torch.device, logit_scale: float) -> Dict[str, float]:
    y_true = []
    y_pred = []
    for batch in loader:
        images = batch["video"].to(device, non_blocking=True)
        labels = batch["label"].long().to(device, non_blocking=True)
        image_features = model(images, None, mode="video")["img_emb"]
        image_features = F.normalize(image_features, dim=-1)
        logits = logit_scale * (image_features @ text_features.T)
        preds = logits.argmax(dim=1)
        y_true.append(labels.cpu().numpy())
        y_pred.append(preds.cpu().numpy())

    if not y_true:
        return {"num_samples": 0, "acc": 0.0, "macro_f1": 0.0}

    y_true_arr = np.concatenate(y_true, axis=0)
    y_pred_arr = np.concatenate(y_pred, axis=0)
    acc, macro_f1 = video_metrics(y_true_arr, y_pred_arr, num_classes=text_features.shape[0])
    return {"num_samples": int(y_true_arr.size), "acc": acc, "macro_f1": macro_f1}


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    cfg = Config.fromfile(args.config)["config"]
    model, _ = surgvlp.load(cfg.model_config, device=device, pretrain=args.checkpoint)
    model = model.to(device)
    model.eval()

    prompts = read_prompts(args.class_prompt)
    tokens = surgvlp.tokenize(prompts, device=device)
    text_features = model(None, tokens, mode="text")["text_emb"]
    text_features = F.normalize(text_features, dim=-1)

    vids = available_vids(args.video_root, args.csv_root)
    per_video = []
    all_true = []
    all_pred = []

    for vid in vids:
        ds_cfg = dict(
            type="Recognition_frame",
            csv_root=args.csv_root,
            vid=f"video_{vid}.csv",
            video_root=args.video_root,
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

        y_true = []
        y_pred = []
        for batch in loader:
            images = batch["video"].to(device, non_blocking=True)
            labels = batch["label"].long().to(device, non_blocking=True)
            image_features = model(images, None, mode="video")["img_emb"]
            image_features = F.normalize(image_features, dim=-1)
            logits = args.logit_scale * (image_features @ text_features.T)
            preds = logits.argmax(dim=1)
            y_true.append(labels.cpu().numpy())
            y_pred.append(preds.cpu().numpy())

        if y_true:
            y_true_arr = np.concatenate(y_true, axis=0)
            y_pred_arr = np.concatenate(y_pred, axis=0)
            acc, macro_f1 = video_metrics(y_true_arr, y_pred_arr, num_classes=text_features.shape[0])
            num_samples = int(y_true_arr.size)
            all_true.append(y_true_arr)
            all_pred.append(y_pred_arr)
        else:
            acc = 0.0
            macro_f1 = 0.0
            num_samples = 0

        per_video.append(
            {
                "vid": vid,
                "num_samples": num_samples,
                "acc": acc,
                "macro_f1": macro_f1,
            }
        )

    avg_acc = float(np.mean([x["acc"] for x in per_video])) if per_video else 0.0
    avg_f1 = float(np.mean([x["macro_f1"] for x in per_video])) if per_video else 0.0

    if all_true:
        y_true_all = np.concatenate(all_true, axis=0)
        y_pred_all = np.concatenate(all_pred, axis=0)
        global_acc, global_macro_f1 = video_metrics(y_true_all, y_pred_all, num_classes=text_features.shape[0])
        total_samples = int(y_true_all.size)
    else:
        global_acc, global_macro_f1, total_samples = 0.0, 0.0, 0

    print(f"[Info] checkpoint={args.checkpoint}")
    print(f"[Info] video_root={args.video_root}")
    print(f"[Info] vids={len(vids)} total_samples={total_samples}")
    print(f"[Result] avg_video_acc={avg_acc:.4f} avg_video_macro_f1={avg_f1:.4f}")
    print(f"[Result] global_acc={global_acc:.4f} global_macro_f1={global_macro_f1:.4f}")

    out = args.output_json
    if not out:
        subset_name = os.path.basename(os.path.normpath(args.video_root))
        ckpt_stem = Path(args.checkpoint).stem
        out = os.path.join(os.path.dirname(args.checkpoint), f"{subset_name}_eval_{ckpt_stem}.json")

    payload = {
        "checkpoint": args.checkpoint,
        "config": args.config,
        "video_root": args.video_root,
        "csv_root": args.csv_root,
        "class_prompt": args.class_prompt,
        "logit_scale": float(args.logit_scale),
        "num_videos": len(vids),
        "num_samples": total_samples,
        "avg_video_acc": avg_acc,
        "avg_video_macro_f1": avg_f1,
        "global_acc": global_acc,
        "global_macro_f1": global_macro_f1,
        "per_video": per_video,
    }

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"[Info] wrote {out}")


if __name__ == "__main__":
    main()
