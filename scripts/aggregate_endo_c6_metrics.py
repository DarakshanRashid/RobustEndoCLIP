#!/usr/bin/env python3
"""Compute Clean accuracy, per-corruption accuracy, Mean-C, and Worst-C from
the JSON files written by scripts/infer.sh.

No script in the original codebase computed Mean-C/Worst-C -- this fills that
gap. Self-contained (stdlib only).

Endo-C6 = {defocus, fog, shot, motion, packet, smoke} only. The six
exploratory corruptions from the original development repo (blood,
brightness, contrast, gaussian_noise, impulse_noise, zoom_blur) are never
folded in, even if such files happen to be present in the eval directory.

Mean-C/Worst-C are reported as MISSING (not 0.0, not a partial average) for
any dataset where fewer than all six Endo-C6 conditions were found.

Usage:
    python scripts/aggregate_endo_c6_metrics.py --eval-dir eval_outputs/robustendoclip_d3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ENDO_C6 = ("defocus", "fog", "shot", "motion", "packet", "smoke")
MISSING = "MISSING"


def classify(filename: str) -> str | None:
    """Return 'clean', one of ENDO_C6, or None (exploratory/unrecognized)."""
    name = filename.lower()
    if name.endswith(".json"):
        name = name[:-5]
    if "clean" in name:
        return "clean"
    for exploratory in ("blood", "brightness", "contrast", "gaussian_noise", "impulse_noise", "zoom_blur"):
        if exploratory in name:
            return None
    checks = [
        ("defocus_blur", "defocus"),
        ("fog", "fog"),
        ("shot_noise", "shot"),
        ("motion_blur", "motion"),
        ("packet_loss", "packet"),
        ("smoke", "smoke"),
    ]
    for substring, label in checks:
        if substring in name:
            return label
    return None


def extract_accuracy(data: dict) -> float | None:
    if "avg_video_acc" in data:
        return float(data["avg_video_acc"])
    metrics = data.get("metrics")
    if isinstance(metrics, dict) and "acc" in metrics:
        return float(metrics["acc"])
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eval-dir", required=True, help="Directory of eval JSONs written by scripts/infer.sh.")
    parser.add_argument(
        "--datasets",
        default="cholect50,kvasir,temset",
        help="Comma-separated dataset name prefixes to look for (default: cholect50,kvasir,temset).",
    )
    args = parser.parse_args()

    eval_dir = Path(args.eval_dir)
    if not eval_dir.is_dir():
        print(f"[Error] Not a directory: {eval_dir}")
        return 1

    datasets = args.datasets.split(",")
    results: dict[str, dict[str, float]] = {ds: {} for ds in datasets}

    for json_file in sorted(eval_dir.glob("*.json")):
        dataset = next((ds for ds in datasets if json_file.name.startswith(f"{ds}_")), None)
        if dataset is None:
            continue
        condition = classify(json_file.name)
        if condition is None:
            continue
        try:
            data = json.loads(json_file.read_text())
        except Exception as exc:  # noqa: BLE001
            print(f"[Warn] could not read {json_file}: {exc}")
            continue
        acc = extract_accuracy(data)
        if acc is None:
            print(f"[Warn] no accuracy field found in {json_file}")
            continue
        if condition in results[dataset]:
            print(f"[Warn] duplicate {dataset}/{condition} found in {json_file}; keeping first value")
            continue
        results[dataset][condition] = acc * 100.0

    header = ["dataset", "clean"] + list(ENDO_C6) + ["mean_c", "worst_c"]
    print(",".join(header))
    for dataset in datasets:
        row = results[dataset]
        clean = row.get("clean")
        corr_values = [row.get(c) for c in ENDO_C6]
        if all(v is not None for v in corr_values):
            mean_c = sum(corr_values) / 6.0
            worst_c = min(corr_values)
        else:
            mean_c = worst_c = None
        cells = [dataset, f"{clean:.2f}" if clean is not None else MISSING]
        cells += [f"{v:.2f}" if v is not None else MISSING for v in corr_values]
        cells += [f"{mean_c:.2f}" if mean_c is not None else MISSING]
        cells += [f"{worst_c:.2f}" if worst_c is not None else MISSING]
        print(",".join(cells))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
