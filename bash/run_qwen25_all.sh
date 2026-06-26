#!/usr/bin/env bash
set -euo pipefail

# End-to-end runs for the Qwen2.5 models:
# 1. FLOAT baseline on cuda:1
# 2. RTN quantized runs on cuda:1
# 3. RBVT quantized runs on cuda:1

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-rbvtquant}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

FLOAT_DEVICE="${FLOAT_DEVICE:-cuda:1}"
RTN_DEVICE="${RTN_DEVICE:-cuda:1}"
RBVT_DEVICE="${RBVT_DEVICE:-cuda:1}"

MODELS=(
  "Qwen/Qwen2.5-7B"
  "Qwen/Qwen2.5-14B"
)

QUANTIZERS=(
  "nf4"
  "nf3"
)

slugify() {
  local s="$1"
  s="${s//\//_}"
  s="${s//./_}"
  s="${s//-/_}"
  echo "$s"
}

build_wandb_args() {
  local -n out_ref=$1
  out_ref=()
  if [ "$USE_WANDB" = "1" ]; then
    out_ref+=(--use-wandb --wandb-project "$WANDB_PROJECT")
    if [ -n "$WANDB_ENTITY" ]; then
      out_ref+=(--wandb-entity "$WANDB_ENTITY")
    fi
  fi
}

for model in "${MODELS[@]}"; do
  model_slug="$(slugify "$model")"
  echo "=== FLOAT / RTN / RBVT runs for ${model} ==="

  WANDB_ARGS=()
  build_wandb_args WANDB_ARGS

  python main.py \
    --model-path "$model" \
    --device "$FLOAT_DEVICE" \
    --method float \
    --output-dir "./outputs/${model_slug}_float" \
    --eval-max-length 2048 \
    --include-lm-eval \
    --lm-eval-task-preset extended \
    "${WANDB_ARGS[@]}"

  for quantizer in "${QUANTIZERS[@]}"; do
    python main.py \
      --model-path "$model" \
      --device "$RTN_DEVICE" \
      --method rtn \
      --quantizer "$quantizer" \
      --output-dir "./outputs/${model_slug}_rtn_${quantizer}" \
      --calib-dataset c4 \
      --max-length 2048 \
      --eval-max-length 2048 \
      --include-lm-eval \
      --lm-eval-task-preset extended \
      "${WANDB_ARGS[@]}"
  done

  for quantizer in "${QUANTIZERS[@]}"; do
    python main.py \
      --model-path "$model" \
      --device "$RBVT_DEVICE" \
      --method rbvt \
      --quantizer "$quantizer" \
      --output-dir "./outputs/${model_slug}_rbvt_${quantizer}" \
      --rbvt-lambda 1.0 \
      --rbvt-topk 0 \
      --calib-dataset c4 \
      --max-length 2048 \
      --eval-max-length 2048 \
      --include-lm-eval \
      --lm-eval-task-preset extended \
      "${WANDB_ARGS[@]}"
  done
done
