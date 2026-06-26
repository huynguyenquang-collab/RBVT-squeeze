#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-rbvtquant}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

MODELS=(
  "meta-llama/Llama-3.1-8B"
  "mistralai/Mistral-7B-v0.3"
  "meta-llama/Meta-Llama-3-8B"
  "Qwen/Qwen2.5-7B"
  "Qwen/Qwen2.5-14B"
)

slugify() {
  local s="$1"
  s="${s//\//_}"
  s="${s//./_}"
  s="${s//-/_}"
  echo "$s"
}

for model in "${MODELS[@]}"; do
  model_slug="$(slugify "$model")"
  WANDB_ARGS=()
  if [ "$USE_WANDB" = "1" ]; then
    WANDB_ARGS+=(--use-wandb --wandb-project "$WANDB_PROJECT")
    if [ -n "$WANDB_ENTITY" ]; then
      WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
    fi
  fi

  python main.py \
    --model-path "$model" \
    --device cuda:0 \
    --method float \
    --output-dir "./outputs/${model_slug}_float" \
    --eval-max-length 2048 \
    --include-lm-eval \
    --lm-eval-task-preset extended \
    "${WANDB_ARGS[@]}"
done
