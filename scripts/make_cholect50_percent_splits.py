#!/usr/bin/env python3
"""Create stratified percentage splits for CholecT50 train pool.

Requested splits:
- train=4% of full train pool, val=1% of remaining pool
- train=8% of full train pool, val=2% of remaining pool
- train=16% of full train pool, val=3% of remaining pool

Data source:
- Per-video CSV files under tests/cholect50/csvs with schema: path,label

Outputs for each split:
- JSON metadata with indices/stats
- train_manifest.txt / val_manifest.txt
- train_csvs/ and val_csvs/ with per-video CSV files in original schema
"""

import argparse
import csv
import json
import os
import random
from collections import defaultdict
from typing import Dict, List, Tuple


CHOLEC_OFFICIAL_TEST_VIDS = {
    "VID06",
    "VID10",
    "VID14",
    "VID32",
    "VID42",
    "VID51",
    "VID73",
    "VID74",
    "VID80",
    "VID111",
}

SPLIT_PAIRS = [(4, 1), (8, 2), (16, 3)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create CholecT50 percentage splits from CSV train pool.")
    parser.add_argument(
        "--csv-root",
        type=str,
        default="/DATA3/SurgVLP-main/tests/cholect50/csvs",
        help="Directory containing video_VID*.csv files.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/DATA3/SurgVLP-main/outputs/splits/cholect50_percent_splits",
        help="Directory to save split outputs.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic split sampling.",
    )
    parser.add_argument(
        "--include-official-test",
        action="store_true",
        help="Include official CholecT50 test videos in the source pool.",
    )
    return parser.parse_args()


def sorted_vids(csv_root: str) -> List[str]:
    vids = []
    for fname in os.listdir(csv_root):
        if fname.startswith("video_") and fname.endswith(".csv"):
            vid = fname[len("video_") : -len(".csv")]
            vids.append(vid)
    vids = sorted(vids, key=lambda x: int(x.replace("VID", "")))
    if not vids:
        raise RuntimeError(f"No video_*.csv files found under {csv_root}")
    return vids


def collect_samples(csv_root: str, include_official_test: bool):
    vids = sorted_vids(csv_root)
    if not include_official_test:
        vids = [v for v in vids if v not in CHOLEC_OFFICIAL_TEST_VIDS]
    if not vids:
        raise RuntimeError("No CholecT50 videos left after filtering.")

    # sample tuple: (csv_file, path, label)
    samples: List[Tuple[str, str, int]] = []
    by_class_indices: Dict[int, List[int]] = defaultdict(list)

    for vid in vids:
        csv_file = f"video_{vid}.csv"
        csv_path = os.path.join(csv_root, csv_file)
        if not os.path.isfile(csv_path):
            continue
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if "path" not in row or "label" not in row:
                    raise RuntimeError(f"Unexpected CSV schema in {csv_path}. Expected columns: path,label")
                path = row["path"].strip()
                label = int(row["label"])
                idx = len(samples)
                samples.append((csv_file, path, label))
                by_class_indices[label].append(idx)

    if not samples:
        raise RuntimeError("No frame samples collected from source CSV files.")
    return vids, samples, by_class_indices


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
        ids = list(by_class_indices[label])
        rng.shuffle(ids)
        n = len(ids)

        train_n = int(round(n * (train_pct / 100.0)))
        if train_pct > 0 and n > 0 and train_n == 0:
            train_n = 1
        train_n = min(max(train_n, 0), n)

        remaining_n = n - train_n
        val_n = int(round(remaining_n * (val_pct_of_remaining / 100.0)))
        if val_pct_of_remaining > 0 and remaining_n > 0 and val_n == 0:
            val_n = 1
        val_n = min(max(val_n, 0), remaining_n)

        train_part = ids[:train_n]
        val_part = ids[train_n : train_n + val_n]

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


def write_manifest(path: str, samples: List[Tuple[str, str, int]], indices: List[int]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for idx in indices:
            csv_file, frame_path, label = samples[idx]
            f.write(f"{csv_file},{frame_path},{label}\n")


def write_split_csvs(root: str, samples: List[Tuple[str, str, int]], indices: List[int]) -> None:
    grouped: Dict[str, List[Tuple[str, int]]] = defaultdict(list)
    for idx in indices:
        csv_file, frame_path, label = samples[idx]
        grouped[csv_file].append((frame_path, label))

    os.makedirs(root, exist_ok=True)
    for csv_file, rows in grouped.items():
        out_path = os.path.join(root, csv_file)
        with open(out_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["path", "label"])
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    vids, samples, by_class = collect_samples(args.csv_root, args.include_official_test)
    total_n = len(samples)
    labels = sorted(by_class.keys())

    print(f"[Info] csv_root={args.csv_root}")
    print(f"[Info] include_official_test={args.include_official_test}")
    print(f"[Info] num_videos={len(vids)}")
    print(f"[Info] total_samples={total_n}")
    print(f"[Info] classes={len(labels)} labels={labels}")
    print(f"[Info] seed={args.seed}")

    for train_pct, val_pct in SPLIT_PAIRS:
        train_idx, val_idx, class_stats = compute_split(
            by_class_indices=by_class,
            train_pct=train_pct,
            val_pct_of_remaining=val_pct,
            seed=args.seed,
        )
        split_name = f"cholect50_train{train_pct}pct_val{val_pct}pct_of_remaining_seed{args.seed}"
        split_dir = os.path.join(args.output_dir, split_name)
        train_csv_root = os.path.join(split_dir, "train_csvs")
        val_csv_root = os.path.join(split_dir, "val_csvs")
        train_manifest = os.path.join(split_dir, "train_manifest.txt")
        val_manifest = os.path.join(split_dir, "val_manifest.txt")
        json_path = os.path.join(split_dir, f"{split_name}.json")

        os.makedirs(split_dir, exist_ok=True)
        write_split_csvs(train_csv_root, samples, train_idx)
        write_split_csvs(val_csv_root, samples, val_idx)
        write_manifest(train_manifest, samples, train_idx)
        write_manifest(val_manifest, samples, val_idx)

        payload = {
            "version": 1,
            "source_csv_root": args.csv_root,
            "include_official_test": bool(args.include_official_test),
            "excluded_official_test_vids": [] if args.include_official_test else sorted(CHOLEC_OFFICIAL_TEST_VIDS),
            "used_vids": vids,
            "seed": int(args.seed),
            "train_percent": int(train_pct),
            "val_percent_of_remaining": int(val_pct),
            "paths": {
                "train_csv_root": train_csv_root,
                "val_csv_root": val_csv_root,
                "train_manifest": train_manifest,
                "val_manifest": val_manifest,
            },
            "dataset": {
                "name": "cholect50",
                "train_indices": train_idx,
                "val_indices": val_idx,
            },
            "stats": {
                "num_videos": len(vids),
                "total_samples": total_n,
                "train_samples": len(train_idx),
                "val_samples": len(val_idx),
                "class_stats": class_stats,
            },
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(
            f"[OK] {split_name}: "
            f"train={len(train_idx)} ({100.0 * len(train_idx) / total_n:.2f}%), "
            f"val={len(val_idx)} ({100.0 * len(val_idx) / total_n:.2f}%), "
            f"dir={split_dir}"
        )


if __name__ == "__main__":
    main()
