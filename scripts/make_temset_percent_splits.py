#!/usr/bin/env python3
"""Create stratified percentage splits for TEMSET training annotations.

Requested splits:
- train=4% of full train.txt, val=1% of remaining train pool
- train=8% of full train.txt, val=2% of remaining train pool
- train=16% of full train.txt, val=3% of remaining train pool

Outputs for each split:
- JSON metadata with indices and stats
- train/val annotation txt files in TEMSET format: "<clip_id> <label>"
"""

import argparse
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple


SPLIT_PAIRS = [(4, 1), (8, 2), (16, 3)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create TEMSET percentage splits from train annotations.")
    parser.add_argument(
        "--train-ann",
        type=str,
        default="/DATA3/tc-clip-main/datasets_splits/temset_splits/train.txt",
        help="Path to full TEMSET training annotation file.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/DATA3/SurgVLP-main/outputs/splits/temset_percent_splits",
        help="Directory to save split files.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split sampling.",
    )
    return parser.parse_args()


def parse_ann_file(path: str) -> List[Tuple[str, int]]:
    samples: List[Tuple[str, int]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) != 2:
                raise ValueError(f"Invalid format at {path}:{line_no}: {line}")
            clip_id, label_str = parts
            samples.append((clip_id, int(label_str)))
    if not samples:
        raise RuntimeError(f"No samples found in {path}")
    return samples


def build_by_class(samples: List[Tuple[str, int]]) -> Dict[int, List[int]]:
    by_class: Dict[int, List[int]] = defaultdict(list)
    for idx, (_, label) in enumerate(samples):
        by_class[label].append(idx)
    return by_class


def compute_split(
    by_class_indices: Dict[int, List[int]],
    train_pct: int,
    val_pct_of_remaining: int,
    seed: int,
) -> Tuple[List[int], List[int], Dict[int, Dict[str, int]]]:
    rng = random.Random(seed * 10_000 + train_pct * 100 + val_pct_of_remaining)
    train_indices: List[int] = []
    val_indices: List[int] = []
    per_class_stats: Dict[int, Dict[str, int]] = {}

    for label in sorted(by_class_indices.keys()):
        cls_indices = list(by_class_indices[label])
        rng.shuffle(cls_indices)
        n = len(cls_indices)

        train_n = int(round(n * (train_pct / 100.0)))
        if train_pct > 0 and n > 0 and train_n == 0:
            train_n = 1
        train_n = min(max(train_n, 0), n)

        remaining_n = n - train_n
        val_n = int(round(remaining_n * (val_pct_of_remaining / 100.0)))
        if val_pct_of_remaining > 0 and remaining_n > 0 and val_n == 0:
            val_n = 1
        val_n = min(max(val_n, 0), remaining_n)

        train_part = cls_indices[:train_n]
        val_part = cls_indices[train_n : train_n + val_n]

        train_indices.extend(train_part)
        val_indices.extend(val_part)

        per_class_stats[label] = {
            "total": n,
            "train": len(train_part),
            "val": len(val_part),
            "remaining_after_train": remaining_n,
        }

    train_set = set(train_indices)
    val_set = set(val_indices)
    if train_set & val_set:
        raise RuntimeError("Internal error: train/val overlap detected.")

    return sorted(train_indices), sorted(val_indices), per_class_stats


def write_ann(path: str, samples: List[Tuple[str, int]], indices: List[int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for idx in indices:
            clip_id, label = samples[idx]
            f.write(f"{clip_id} {label}\n")


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    samples = parse_ann_file(args.train_ann)
    by_class = build_by_class(samples)

    total_n = len(samples)
    labels = sorted(by_class.keys())
    print(f"[Info] train_ann={args.train_ann}")
    print(f"[Info] total_samples={total_n}")
    print(f"[Info] classes={len(labels)}")
    print(f"[Info] seed={args.seed}")

    for train_pct, val_pct in SPLIT_PAIRS:
        train_idx, val_idx, class_stats = compute_split(
            by_class_indices=by_class,
            train_pct=train_pct,
            val_pct_of_remaining=val_pct,
            seed=args.seed,
        )
        split_name = f"temset_train{train_pct}pct_val{val_pct}pct_of_remaining_seed{args.seed}"

        json_path = os.path.join(args.output_dir, f"{split_name}.json")
        train_ann_path = os.path.join(args.output_dir, f"{split_name}_train.txt")
        val_ann_path = os.path.join(args.output_dir, f"{split_name}_val.txt")

        payload = {
            "version": 1,
            "source_train_ann": args.train_ann,
            "seed": int(args.seed),
            "train_percent": int(train_pct),
            "val_percent_of_remaining": int(val_pct),
            "train_ann_file": train_ann_path,
            "val_ann_file": val_ann_path,
            "dataset": {
                "name": "temset",
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

        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        write_ann(train_ann_path, samples, train_idx)
        write_ann(val_ann_path, samples, val_idx)

        print(
            f"[OK] {split_name}: "
            f"train={len(train_idx)} ({100.0 * len(train_idx) / total_n:.2f}%), "
            f"val={len(val_idx)} ({100.0 * len(val_idx) / total_n:.2f}%), "
            f"json={json_path}"
        )


if __name__ == "__main__":
    main()
