#!/usr/bin/env python3
"""Create stratified percentage splits for Kvasir.

Requested splits:
- train=4% of full data, val=1% of remaining train pool
- train=8% of full data, val=2% of remaining train pool
- train=16% of full data, val=3% of remaining train pool

Outputs for each split:
- JSON compatible with train_surgvlp_vera_kvasir_4shot.py (--split-file)
- train/val manifest text files (rel_path label)
"""

import argparse
import json
import os
import random
from typing import Dict, List, Tuple


CLASS_ORDER = [
    "dyed-lifted-polyps",
    "dyed-resection-margins",
    "esophagitis",
    "normal-cecum",
    "normal-pylorus",
    "normal-z-line",
    "polyps",
    "ulcerative-colitis",
]

IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
SPLIT_PAIRS = [(4, 1), (8, 2), (16, 3)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Kvasir percentage splits.")
    parser.add_argument(
        "--root",
        type=str,
        default="/DATA3/TDA/kvasir-dataset-v2",
        help="Kvasir dataset root.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/DATA3/SurgVLP-main/outputs/splits/kvasir_percent_splits",
        help="Directory to save split files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split sampling.",
    )
    return parser.parse_args()


def collect_samples(root: str) -> Tuple[List[Tuple[str, int, str]], Dict[str, List[int]]]:
    """Collect samples in the same class-order/style as training scripts."""
    all_samples: List[Tuple[str, int, str]] = []
    by_class_indices: Dict[str, List[int]] = {c: [] for c in CLASS_ORDER}

    for cls_idx, cls_name in enumerate(CLASS_ORDER):
        cls_dir = os.path.join(root, cls_name)
        if not os.path.isdir(cls_dir):
            raise RuntimeError(f"Class directory missing: {cls_dir}")

        file_names = sorted(
            [f for f in os.listdir(cls_dir) if f.lower().endswith(IMAGE_EXTS)]
        )
        if not file_names:
            raise RuntimeError(f"No images found in {cls_dir}")

        for fname in file_names:
            abs_path = os.path.join(cls_dir, fname)
            rel_path = os.path.join(cls_name, fname)
            sample_idx = len(all_samples)
            all_samples.append((abs_path, cls_idx, rel_path))
            by_class_indices[cls_name].append(sample_idx)

    return all_samples, by_class_indices


def compute_split(
    by_class_indices: Dict[str, List[int]],
    train_pct: int,
    val_pct_of_remaining: int,
    seed: int,
) -> Tuple[List[int], List[int], Dict[str, Dict[str, int]]]:
    rng = random.Random(seed * 10_000 + train_pct * 100 + val_pct_of_remaining)
    train_indices: List[int] = []
    val_indices: List[int] = []
    per_class_stats: Dict[str, Dict[str, int]] = {}

    for cls_name in CLASS_ORDER:
        cls_indices = list(by_class_indices[cls_name])
        rng.shuffle(cls_indices)
        n = len(cls_indices)

        train_n = int(round(n * (train_pct / 100.0)))
        train_n = max(1, min(train_n, n))

        remaining_n = n - train_n
        val_n = int(round(remaining_n * (val_pct_of_remaining / 100.0)))
        if remaining_n > 0 and val_pct_of_remaining > 0 and val_n == 0:
            val_n = 1
        val_n = min(max(val_n, 0), remaining_n)

        train_part = cls_indices[:train_n]
        val_part = cls_indices[train_n : train_n + val_n]

        train_indices.extend(train_part)
        val_indices.extend(val_part)

        per_class_stats[cls_name] = {
            "total": n,
            "train": len(train_part),
            "val": len(val_part),
            "remaining_after_train": remaining_n,
        }

    return sorted(train_indices), sorted(val_indices), per_class_stats


def write_manifest(path: str, all_samples: List[Tuple[str, int, str]], indices: List[int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for idx in indices:
            _, label, rel_path = all_samples[idx]
            f.write(f"{rel_path} {label}\n")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    all_samples, by_class_indices = collect_samples(args.root)
    total_n = len(all_samples)
    print(f"[Info] root={args.root}")
    print(f"[Info] total_samples={total_n}")
    print(f"[Info] classes={len(CLASS_ORDER)}")
    print(f"[Info] seed={args.seed}")

    for train_pct, val_pct in SPLIT_PAIRS:
        train_idx, val_idx, class_stats = compute_split(
            by_class_indices=by_class_indices,
            train_pct=train_pct,
            val_pct_of_remaining=val_pct,
            seed=args.seed,
        )
        split_name = f"kvasir_train{train_pct}pct_val{val_pct}pct_of_remaining_seed{args.seed}"

        payload = {
            "version": 1,
            "root": args.root,
            "seed": int(args.seed),
            "train_percent": int(train_pct),
            "val_percent_of_remaining": int(val_pct),
            "dataset": {
                "name": "kvasir",
                "train_indices": train_idx,
                "val_indices": val_idx,
            },
            "stats": {
                "total_samples": total_n,
                "train_samples": len(train_idx),
                "val_samples": len(val_idx),
                "class_stats": class_stats,
            },
        }

        json_path = os.path.join(args.output_dir, f"{split_name}.json")
        train_manifest = os.path.join(args.output_dir, f"{split_name}_train.txt")
        val_manifest = os.path.join(args.output_dir, f"{split_name}_val.txt")

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        write_manifest(train_manifest, all_samples, train_idx)
        write_manifest(val_manifest, all_samples, val_idx)

        print(
            f"[OK] {split_name}: "
            f"train={len(train_idx)} ({100.0 * len(train_idx) / total_n:.2f}%), "
            f"val={len(val_idx)} ({100.0 * len(val_idx) / total_n:.2f}%), "
            f"json={json_path}"
        )


if __name__ == "__main__":
    main()
