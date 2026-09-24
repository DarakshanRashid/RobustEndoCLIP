#!/usr/bin/env bash
# Train RobustEndoCLIP (VeRA): SurgVLP with frozen encoders + VeRA on the
# final visual projection (backbone_img.global_embedder), for one label
# budget. This is the training command for Table 1's RobustEndoCLIP row and
# Table 2's VeRA row.
#
# Internally chains three calls to train_robustendoclip_vera.py -- kvasir ->
# cholect50 -> temset -- passing each stage's output checkpoint as the next
# stage's --pretrain, the same convention used throughout this codebase.
#
# STATUS: RECONSTRUCTED, not a verified command. No wrapper script or log for
# ANY VeRA training stage survived in the original development repository.
# This chaining is a best-effort reconstruction from:
#   - train_robustendoclip_vera.py's own argparse defaults
#   - the exact epoch counts observed in the original run's saved logs
#     (Kvasir stage: 100 epochs -- double the paper's stated 50; CholecT50
#     and TEMSET-24K stages: 50 epochs each, matching the paper)
#   - hyperparameters matching the paper's stated Adam / lr 1e-3 / batch 16
# The --pretrain chaining itself (does stage 2 really start from stage 1's
# output?) is inferred only from checkpoint-naming convention and clean
# sequential timestamps in the original logs -- not proven from a surviving
# command. See REPRODUCE.md for the full accounting.
#
# Usage:
#   SPLIT_TAG=d3 \
#   BASE_CHECKPOINT=<path to SurgVLP.pth> \
#   KVASIR_ROOT=<path> \
#   CHOLECT50_ROOT=<path> CHOLECT50_TRAIN_CSV_ROOT=<path> CHOLECT50_VAL_CSV_ROOT=<path> \
#   TEMSET_ROOT=<path> TEMSET_TRAIN_ANN=<path> TEMSET_VAL_ANN=<path> \
#   scripts/train_robustendoclip_vera.sh
#
# SPLIT_TAG: d1 = 4% label budget, d2 = 8%, d3 = 16%.
# The CHOLECT50_*_CSV_ROOT and TEMSET_*_ANN paths are produced by
# make_cholect50_percent_splits.py / make_temset_percent_splits.py for the
# same SPLIT_TAG -- run those first.
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
SPLIT_TAG="${SPLIT_TAG:-d3}"

case "$SPLIT_TAG" in
  d1) BUDGET_PCT=4 ;;
  d2) BUDGET_PCT=8 ;;
  d3) BUDGET_PCT=16 ;;
  *) echo "[Error] SPLIT_TAG must be d1 (4%), d2 (8%), or d3 (16%) -- got '$SPLIT_TAG'" >&2; exit 1 ;;
esac

BASE_CHECKPOINT="${BASE_CHECKPOINT:?Set BASE_CHECKPOINT to the base SurgVLP.pth checkpoint path.}"
KVASIR_ROOT="${KVASIR_ROOT:?Set KVASIR_ROOT to the full Kvasir-v2 dataset root (8 classes x 1000 images).}"
CHOLECT50_ROOT="${CHOLECT50_ROOT:?Set CHOLECT50_ROOT to the clean CholecT50 frame root (e.g. .../CholecT50/data).}"
CHOLECT50_TRAIN_CSV_ROOT="${CHOLECT50_TRAIN_CSV_ROOT:?Set CHOLECT50_TRAIN_CSV_ROOT to the split's train_csvs/ dir from make_cholect50_percent_splits.py.}"
CHOLECT50_VAL_CSV_ROOT="${CHOLECT50_VAL_CSV_ROOT:?Set CHOLECT50_VAL_CSV_ROOT to the split's val_csvs/ dir from make_cholect50_percent_splits.py.}"
TEMSET_ROOT="${TEMSET_ROOT:?Set TEMSET_ROOT to the TEMSET-24K training frame root.}"
TEMSET_TRAIN_ANN="${TEMSET_TRAIN_ANN:?Set TEMSET_TRAIN_ANN to the split's train annotation .txt from make_temset_percent_splits.py.}"
TEMSET_VAL_ANN="${TEMSET_VAL_ANN:?Set TEMSET_VAL_ANN to the split's val annotation .txt from make_temset_percent_splits.py.}"

OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/checkpoints}"
mkdir -p "$OUTPUT_DIR"

VERA_ARGS=(--vera-rank 256 --vera-dropout 0.05 --vera-d-initial 0.1 --batch-size 16 --eval-batch-size 128 --lr 1e-3 --weight-decay 0.0)

# Intermediate (unbranded) stage checkpoints; only the final one is "the"
# released RobustEndoCLIP VeRA checkpoint for this label budget.
KVASIR_CKPT="$OUTPUT_DIR/.stage1_kvasir_${BUDGET_PCT}pct.pth"
CHOLECT50_CKPT="$OUTPUT_DIR/.stage2_cholect50_${BUDGET_PCT}pct.pth"
FINAL_CKPT="$OUTPUT_DIR/RobustEndoCLIP_VeRA_${BUDGET_PCT}pct.pth"

echo "[Stage 1/3] kvasir (100 epochs) -> $KVASIR_CKPT"
"$PY" "$ROOT_DIR/scripts/train_robustendoclip_vera.py" \
  --dataset kvasir --split-tag "$SPLIT_TAG" \
  --config "$ROOT_DIR/tests/config_surgvlp.py" \
  --pretrain "$BASE_CHECKPOINT" \
  --kvasir-root "$KVASIR_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --save-model "$KVASIR_CKPT" \
  --log-csv "$OUTPUT_DIR/train_log_kvasir_${BUDGET_PCT}pct.csv" \
  --epochs 100 "${VERA_ARGS[@]}"

echo "[Stage 2/3] cholect50 (50 epochs) -> $CHOLECT50_CKPT"
"$PY" "$ROOT_DIR/scripts/train_robustendoclip_vera.py" \
  --dataset cholect50 --split-tag "$SPLIT_TAG" \
  --config "$ROOT_DIR/tests/config_surgvlp.py" \
  --pretrain "$KVASIR_CKPT" \
  --cholec-video-root "$CHOLECT50_ROOT" \
  --cholect50-train-csv-root "$CHOLECT50_TRAIN_CSV_ROOT" \
  --cholect50-val-csv-root "$CHOLECT50_VAL_CSV_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --save-model "$CHOLECT50_CKPT" \
  --log-csv "$OUTPUT_DIR/train_log_cholect50_${BUDGET_PCT}pct.csv" \
  --epochs 50 "${VERA_ARGS[@]}"

echo "[Stage 3/3] temset (50 epochs) -> $FINAL_CKPT"
"$PY" "$ROOT_DIR/scripts/train_robustendoclip_vera.py" \
  --dataset temset --split-tag "$SPLIT_TAG" \
  --config "$ROOT_DIR/tests/config_surgvlp.py" \
  --pretrain "$CHOLECT50_CKPT" \
  --temset-video-root "$TEMSET_ROOT" \
  --temset-train-ann "$TEMSET_TRAIN_ANN" \
  --temset-val-ann "$TEMSET_VAL_ANN" \
  --output-dir "$OUTPUT_DIR" \
  --save-model "$FINAL_CKPT" \
  --log-csv "$OUTPUT_DIR/train_log_temset_${BUDGET_PCT}pct.csv" \
  --epochs 50 "${VERA_ARGS[@]}"

echo
echo "[Done] RobustEndoCLIP VeRA (${BUDGET_PCT}% label budget) checkpoint: $FINAL_CKPT"
echo "Evaluate it with: CHECKPOINT=$FINAL_CKPT scripts/infer.sh"
