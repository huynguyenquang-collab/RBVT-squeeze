#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
  elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="$(command -v python || command -v python3 || true)"
  fi
fi
LM_EVAL_TASKS="${LM_EVAL_TASKS:-arc_challenge arc_easy boolq hellaswag lambada_openai openbookqa piqa rte winogrande mmlu gsm8k}"

read -r -a LM_EVAL_TASK_ARRAY <<< "$LM_EVAL_TASKS"

"$PYTHON_BIN" lm_eval_smoke.py \
  --model-path "${LM_EVAL_SMOKE_MODEL_PATH:-sshleifer/tiny-gpt2}" \
  --device "${LM_EVAL_SMOKE_DEVICE:-cuda:0}" \
  --tasks "${LM_EVAL_TASK_ARRAY[@]}" \
  --limit "${LM_EVAL_SMOKE_LIMIT:-1}" \
  --batch-size "${LM_EVAL_SMOKE_BATCH_SIZE:-1}" \
  --num-fewshot "${LM_EVAL_SMOKE_NUM_FEWSHOT:-0}"
