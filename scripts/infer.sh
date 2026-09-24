#!/usr/bin/env bash
# Evaluate ONE checkpoint (zero-shot base model, RobustEndoCLIP/VeRA, or LoRA — any
# full state_dict in the SurgVLP format) on clean + all six Endo-C6 corruptions,
# across CholecT50, Kvasir, and TEMSET-24K.
#
# This is the general-purpose evaluation entry point: every row of Table 1 and
# every VeRA/LoRA row of Table 2 in the paper was produced by pointing this
# pattern at a different --checkpoint (see REPRODUCE.md for exactly which
# checkpoint/dataset-root combination reproduces which row).
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
CHECKPOINT="${CHECKPOINT:?Set CHECKPOINT to the .pth file to evaluate (base pretrained, VeRA-merged, or LoRA-merged). See REPRODUCE.md.}"
CONFIG="${CONFIG:-$ROOT_DIR/tests/config_surgvlp.py}"
OUTPUT_DIR="${OUTPUT_DIR:-$ROOT_DIR/eval_outputs}"
DEVICE="${DEVICE:-auto}"
BATCH_SIZE="${BATCH_SIZE:-128}"
NUM_WORKERS="${NUM_WORKERS:-4}"

# Dataset roots. CHOLECT50_CORR_ROOT / KVASIR_CORR_ROOT / TEMSET_CORR_ROOT must
# each contain the six Endo-C6 corruption subdirectories (see REPRODUCE.md for
# the exact expected subfolder names and the data download link).
CHOLECT50_ROOT="${CHOLECT50_ROOT:?Set CHOLECT50_ROOT to your clean CholecT50 frame root (e.g. .../CholecT50/data). See REPRODUCE.md.}"
CHOLECT50_CSV_ROOT="${CHOLECT50_CSV_ROOT:?Set CHOLECT50_CSV_ROOT to the per-video label CSV directory produced from CholecT50 (see commands.txt-equivalent conversion step in REPRODUCE.md).}"
CHOLECT50_CORR_ROOT="${CHOLECT50_CORR_ROOT:?Set CHOLECT50_CORR_ROOT to the root containing the six Endo-C6 CholecT50 corruption subdirectories. See REPRODUCE.md.}"

KVASIR_ROOT="${KVASIR_ROOT:?Set KVASIR_ROOT to your clean Kvasir test root (150 images/class, 8 classes). See REPRODUCE.md.}"
KVASIR_CORR_ROOT="${KVASIR_CORR_ROOT:?Set KVASIR_CORR_ROOT to the root containing the six Endo-C6 Kvasir corruption subdirectories. See REPRODUCE.md.}"

TEMSET_ROOT="${TEMSET_ROOT:?Set TEMSET_ROOT to your clean TEMSET-24K eval frame root. See REPRODUCE.md.}"
TEMSET_ANN_ROOT="${TEMSET_ANN_ROOT:?Set TEMSET_ANN_ROOT to the directory holding the TEMSET clean/corruption annotation .txt files (existing-frame filtered lists). See REPRODUCE.md.}"
TEMSET_CORR_ROOT="${TEMSET_CORR_ROOT:?Set TEMSET_CORR_ROOT to the root containing the TEMSET Endo-C6 corruption subdirectories. See REPRODUCE.md.}"

usage() {
  cat <<EOF
Usage:
  CHECKPOINT=<path> CHOLECT50_ROOT=<path> CHOLECT50_CSV_ROOT=<path> CHOLECT50_CORR_ROOT=<path> \\
  KVASIR_ROOT=<path> KVASIR_CORR_ROOT=<path> \\
  TEMSET_ROOT=<path> TEMSET_ANN_ROOT=<path> TEMSET_CORR_ROOT=<path> \\
  $(basename "$0") [--checkpoint <path>] [--config <path>] [--output-dir <path>]
                    [--device <auto|cuda|cpu>] [--batch-size <int>] [--num-workers <int>]

All dataset-root variables are required (no baked-in default paths) and can
also be set as environment variables instead of flags. See REPRODUCE.md for
the full walkthrough, including which checkpoint reproduces which paper table
row.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint) CHECKPOINT="$2"; shift 2 ;;
    --config) CONFIG="$2"; shift 2 ;;
    --output-dir) OUTPUT_DIR="$2"; shift 2 ;;
    --device) DEVICE="$2"; shift 2 ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --num-workers) NUM_WORKERS="$2"; shift 2 ;;
    --python) PYTHON_BIN="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "[Error] Unknown argument: $1" >&2; usage; exit 1 ;;
  esac
done

require_path() {
  if [[ ! -e "$1" ]]; then
    echo "[Error] Missing required path: $1" >&2
    exit 1
  fi
}

run_cmd() {
  echo
  echo "[Run] $*"
  "$@"
}

mkdir -p "$OUTPUT_DIR"
require_path "$CHECKPOINT"
require_path "$CONFIG"

# --- CholecT50: clean + Endo-C6 (5-video subset VID06/10/14/51/73, matching
#     every candidate result found in the original experiments) ---
CHOLECT50_PROMPTS="$ROOT_DIR/tests/class_prompt.txt"
require_path "$CHOLECT50_CSV_ROOT"
require_path "$CHOLECT50_PROMPTS"

declare -a CHOLECT50_EVAL_ROOTS=(
  "$CHOLECT50_ROOT"
  "$CHOLECT50_CORR_ROOT/defocus_blur_s5_p100"
  "$CHOLECT50_CORR_ROOT/fog_s5_p100"
  "$CHOLECT50_CORR_ROOT/shot_noise_s5_p100"
  "$CHOLECT50_CORR_ROOT/motion_blur_p100_s5"
  "$CHOLECT50_CORR_ROOT/packet_loss_p100_i5"
  "$CHOLECT50_CORR_ROOT/smoke_p100_s5"
)
for root in "${CHOLECT50_EVAL_ROOTS[@]}"; do
  require_path "$root"
  subset="$(basename "$root")"
  [[ "$root" == "$CHOLECT50_ROOT" ]] && subset="clean"
  out_json="${OUTPUT_DIR}/cholect50_${subset}.json"
  run_cmd "$PYTHON_BIN" "$ROOT_DIR/scripts/eval_cholect50_full_checkpoint.py" \
    --checkpoint "$CHECKPOINT" --config "$CONFIG" \
    --video-root "$root" --csv-root "$CHOLECT50_CSV_ROOT" --class-prompt "$CHOLECT50_PROMPTS" \
    --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" --device "$DEVICE" \
    --logit-scale 100.0 --output-json "$out_json"
done

# --- Kvasir: clean + Endo-C6 ---
KVASIR_PROMPTS="$ROOT_DIR/tests/class_prompt_kvasir.txt"
require_path "$KVASIR_PROMPTS"

declare -a KVASIR_EVAL_ROOTS=(
  "$KVASIR_ROOT"
  "$KVASIR_CORR_ROOT/defocus_blur_s5_p100"
  "$KVASIR_CORR_ROOT/fog_s5_p100"
  "$KVASIR_CORR_ROOT/shot_noise_s5_p100"
  "$KVASIR_CORR_ROOT/motion_blur_p100_s5"
  "$KVASIR_CORR_ROOT/packet_loss_p100_i5"
  "$KVASIR_CORR_ROOT/smoke_p100_s5"
)
for root in "${KVASIR_EVAL_ROOTS[@]}"; do
  require_path "$root"
  subset="$(basename "$root")"
  [[ "$root" == "$KVASIR_ROOT" ]] && subset="clean"
  out_json="${OUTPUT_DIR}/kvasir_${subset}.json"
  run_cmd "$PYTHON_BIN" "$ROOT_DIR/scripts/eval_kvasir_full_checkpoint.py" \
    --checkpoint "$CHECKPOINT" --config "$CONFIG" \
    --kvasir-root "$root" --kvasir-prompts "$KVASIR_PROMPTS" \
    --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" --device "$DEVICE" \
    --logit-scale 20.0 --output-json "$out_json"
done

# --- TEMSET-24K: clean + defocus/fog/shot_noise/motion/packet/smoke ---
TEMSET_PROMPTS="$ROOT_DIR/tests/class_prompt_temset.txt"
require_path "$TEMSET_PROMPTS"
require_path "$TEMSET_ANN_ROOT"

declare -a TEMSET_TRIPLES=(
  "clean|$TEMSET_ROOT|$TEMSET_ANN_ROOT/clean_existing.txt"
  "defocus_blur_s5_p100|$TEMSET_CORR_ROOT/defocus_blur_s5_p100|$TEMSET_ANN_ROOT/defocus_blur_s5_p100_existing.txt"
  "fog_s5_p100|$TEMSET_CORR_ROOT/fog_s5_p100|$TEMSET_ANN_ROOT/fog_s5_p100_existing.txt"
  "shot_noise_s5_p100|$TEMSET_CORR_ROOT/shot_noise_s5_p100|$TEMSET_ANN_ROOT/shot_noise_s5_p100_existing.txt"
  "motion_blur_p100_s5|$TEMSET_CORR_ROOT/motion_blur_p100_s5|$TEMSET_ANN_ROOT/motion_blur_p100_s5_existing.txt"
  "packet_loss_p100_i5|$TEMSET_CORR_ROOT/packet_loss_p100_i5|$TEMSET_ANN_ROOT/packet_loss_p100_i5_existing.txt"
  "smoke_p100_s5|$TEMSET_CORR_ROOT/smoke_p100_s5|$TEMSET_ANN_ROOT/smoke_p100_s5_existing.txt"
)
for triple in "${TEMSET_TRIPLES[@]}"; do
  IFS="|" read -r name root ann_file <<<"$triple"
  require_path "$root"
  require_path "$ann_file"
  out_json="${OUTPUT_DIR}/temset_${name}.json"
  run_cmd "$PYTHON_BIN" "$ROOT_DIR/scripts/eval_temset_full_checkpoint.py" \
    --checkpoint "$CHECKPOINT" --config "$CONFIG" \
    --ann-file "$ann_file" --video-root "$root" --class-prompt "$TEMSET_PROMPTS" \
    --batch-size "$BATCH_SIZE" --num-workers "$NUM_WORKERS" --device "$DEVICE" \
    --logit-scale 20.0 --output-json "$out_json"
done

echo
echo "[Done] Evaluation complete. JSON files written to: ${OUTPUT_DIR}"
