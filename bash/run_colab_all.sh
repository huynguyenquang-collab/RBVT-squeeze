#!/usr/bin/env bash
set -euo pipefail

# One-command Colab entrypoint after cloning the repository.
#
# Required before running:
# - select a GPU runtime;
# - mount Google Drive when USE_DRIVE=1;
# - export HF_TOKEN for the gated Llama-3.1-8B model.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_SETUP="${RUN_SETUP:-0}"
RUN_TESTS="${RUN_TESTS:-0}"
RUN_SQUEEZE_FIRST="${RUN_SQUEEZE_FIRST:-1}"
RUN_LEANQUANT_AFTER="${RUN_LEANQUANT_AFTER:-1}"

SQUEEZE_CODEBOOKS="${SQUEEZE_CODEBOOKS:-squeezellm}"
LEANQUANT_CODEBOOKS="${LEANQUANT_CODEBOOKS:-leanquant}"

echo "=== RBVTQuant Colab end-to-end runner ==="
echo "Repository: $ROOT_DIR"

# Limit OpenMP threads to prevent "Thread creation failed" errors during Hessian collection
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
echo "Thread limits: OMP_NUM_THREADS=$OMP_NUM_THREADS"

if [ "$RUN_SETUP" = "1" ]; then
  bash bash/setup_colab_codebooks.sh
fi

if [ "$RUN_TESTS" = "1" ]; then
  bash bash/test_colab_codebooks.sh
fi

# run_colab_codebooks.sh performs preflight checks by default.
if [ "$RUN_SQUEEZE_FIRST" = "1" ]; then
  echo
  echo "=== Phase 1: SqueezeLLM runs ==="
  CODEBOOKS="$SQUEEZE_CODEBOOKS" \
  bash bash/run_colab_codebooks.sh
fi

if [ "$RUN_LEANQUANT_AFTER" = "1" ]; then
  echo
  echo "=== Phase 2: LeanQuant runs ==="
  CODEBOOKS="$LEANQUANT_CODEBOOKS" \
  bash bash/run_colab_codebooks.sh
fi
