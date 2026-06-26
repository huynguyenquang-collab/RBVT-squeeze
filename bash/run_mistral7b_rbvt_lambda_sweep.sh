#!/usr/bin/env bash
set -euo pipefail

# Sweep RBVT lambda on Mistral-7B-v0.3.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-rbvtquant}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

MODEL="${MODEL:-mistralai/Mistral-7B-v0.3}"
DEVICE="${DEVICE:-cuda:0}"
RBVT_TOPK="${RBVT_TOPK:-0}"

QUANTIZERS=(
  "nf4"
  "nf3"
)

LAMBDAS=(
  "0.1"
  "0.5"
  "1.0"
  "3.0"
  "10.0"
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

model_slug="$(slugify "$MODEL")"
WANDB_ARGS=()
build_wandb_args WANDB_ARGS

echo "=== RBVT lambda sweep for ${MODEL} ==="

for quantizer in "${QUANTIZERS[@]}"; do
  for rbvt_lambda in "${LAMBDAS[@]}"; do
    lambda_slug="${rbvt_lambda//./p}"
    python main.py \
      --model-path "$MODEL" \
      --device "$DEVICE" \
      --method rbvt \
      --quantizer "$quantizer" \
      --output-dir "./outputs/${model_slug}_rbvt_${quantizer}_lambda_${lambda_slug}" \
      --rbvt-lambda "$rbvt_lambda" \
      --rbvt-topk "$RBVT_TOPK" \
      --calib-dataset c4 \
      --max-length 2048 \
      --eval-max-length 2048 \
      --include-lm-eval \
      --lm-eval-task-preset extended \
      "${WANDB_ARGS[@]}"
  done
done
