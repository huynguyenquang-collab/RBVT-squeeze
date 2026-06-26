#!/usr/bin/env bash
set -euo pipefail

# Fast adapter and CLI checks. This does not download or quantize Llama-3.1-8B.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-colab}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Error: Python environment not found at $PYTHON_BIN." >&2
  echo "Run: bash bash/setup_colab_codebooks.sh" >&2
  exit 1
fi

echo "=== Codebook adapter tests ==="
"$PYTHON_BIN" -m unittest -v test_codebook_adapters.py

echo
echo "=== Benchmark CLI check ==="
"$PYTHON_BIN" codebook_benchmark.py --help >/dev/null

echo
echo "=== lm-eval datasets compatibility check ==="
"$PYTHON_BIN" lm_eval_dataset_smoke.py

if [ -n "${LM_EVAL_TASKS:-}" ]; then
  echo
  echo "=== lm-eval task smoke test ==="
  read -r -a LM_EVAL_TASK_ARRAY <<< "$LM_EVAL_TASKS"
  "$PYTHON_BIN" lm_eval_smoke.py \
    --model-path "${LM_EVAL_SMOKE_MODEL_PATH:-sshleifer/tiny-gpt2}" \
    --device "${LM_EVAL_SMOKE_DEVICE:-cpu}" \
    --tasks "${LM_EVAL_TASK_ARRAY[@]}" \
    --limit "${LM_EVAL_SMOKE_LIMIT:-1}" \
    --batch-size "${LM_EVAL_SMOKE_BATCH_SIZE:-1}"
fi

echo "Smoke tests passed."
