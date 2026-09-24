#!/usr/bin/env bash
# Train RobustEndoCLIP (LoRA): SurgVLP with frozen encoders + LoRA on the
# final visual projection (backbone_img.global_embedder) -- the ablation
# baseline compared against VeRA in Table 2. 16% label budget only; no d1/d2
# LoRA checkpoint exists in the original results.
#
# Internally chains three calls to train_robustendoclip_lora.py -- kvasir ->
# cholect50 -> temset -- same convention as train_robustendoclip_vera.sh.
#
# STATUS: stage 1 (Kvasir) is a VERIFIED command -- its exact wrapper script
# survived in the original development repository. Stages 2-3 are
# RECONSTRUCTED the same way as the VeRA chain (script defaults + observed
# epoch counts in train_log_*.csv; --pretrain chaining inferred from
# checkpoint-naming convention and sequential timestamps, not proven). See
# REPRODUCE.md for the full accounting.
#
# Usage:
#   BASE_CHECKPOINT=<path to SurgVLP.pth> \
#   KVASIR_ROOT=<path> \
#   CHOLECT50_ROOT=<path> CHOLECT50_TRAIN_CSV_ROOT=<path> CHOLECT50_VAL_CSV_ROOT=<path> \
#   TEMSET_ROOT=<path> TEMSET_TRAIN_ANN=<path> TEMSET_VAL_ANN=<path> \
#   scripts/train_robustendoclip_lora.sh
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
SPLIT_TAG="d3"

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

LORA_ARGS=(--lora-rank 256 --lora-alpha 256 --lora-dropout 0.05 --batch-size 16 --eval-batch-size 128 --lr 1e-3 --weight-decay 0.0)

KVASIR_CKPT="$OUTPUT_DIR/.stage1_kvasir_lora_16pct.pth"
CHOLECT50_CKPT="$OUTPUT_DIR/.stage2_cholect50_lora_16pct.pth"
FINAL_CKPT="$OUTPUT_DIR/RobustEndoCLIP_LoRA_16pct.pth"

echo "[Stage 1/3] kvasir (100 epochs, VERIFIED command) -> $KVASIR_CKPT"
"$PY" "$ROOT_DIR/scripts/train_robustendoclip_lora.py" \
  --dataset kvasir --split-tag "$SPLIT_TAG" \
  --config "$ROOT_DIR/tests/config_surgvlp.py" \
  --pretrain "$BASE_CHECKPOINT" \
  --kvasir-root "$KVASIR_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --save-model "$KVASIR_CKPT" \
  --log-csv "$OUTPUT_DIR/train_log_kvasir_lora_16pct.csv" \
  --epochs 100 "${LORA_ARGS[@]}"

echo "[Stage 2/3] cholect50 (50 epochs, reconstructed) -> $CHOLECT50_CKPT"
"$PY" "$ROOT_DIR/scripts/train_robustendoclip_lora.py" \
  --dataset cholect50 --split-tag "$SPLIT_TAG" \
  --config "$ROOT_DIR/tests/config_surgvlp.py" \
  --pretrain "$KVASIR_CKPT" \
  --cholec-video-root "$CHOLECT50_ROOT" \
  --cholect50-train-csv-root "$CHOLECT50_TRAIN_CSV_ROOT" \
  --cholect50-val-csv-root "$CHOLECT50_VAL_CSV_ROOT" \
  --output-dir "$OUTPUT_DIR" \
  --save-model "$CHOLECT50_CKPT" \
  --log-csv "$OUTPUT_DIR/train_log_cholect50_lora_16pct.csv" \
  --epochs 50 "${LORA_ARGS[@]}"

echo "[Stage 3/3] temset (50 epochs, reconstructed) -> $FINAL_CKPT"
"$PY" "$ROOT_DIR/scripts/train_robustendoclip_lora.py" \
  --dataset temset --split-tag "$SPLIT_TAG" \
  --config "$ROOT_DIR/tests/config_surgvlp.py" \
  --pretrain "$CHOLECT50_CKPT" \
  --temset-video-root "$TEMSET_ROOT" \
  --temset-train-ann "$TEMSET_TRAIN_ANN" \
  --temset-val-ann "$TEMSET_VAL_ANN" \
  --output-dir "$OUTPUT_DIR" \
  --save-model "$FINAL_CKPT" \
  --log-csv "$OUTPUT_DIR/train_log_temset_lora_16pct.csv" \
  --epochs 50 "${LORA_ARGS[@]}"

echo
echo "[Done] RobustEndoCLIP LoRA (16% label budget) checkpoint: $FINAL_CKPT"
echo "Evaluate it with: CHECKPOINT=$FINAL_CKPT scripts/infer.sh"
