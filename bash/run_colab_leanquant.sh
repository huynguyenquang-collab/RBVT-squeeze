#!/usr/bin/env bash
set -euo pipefail

# Colab entrypoint for LeanQuant-only benchmark runs.
# This script intentionally excludes SqueezeLLM jobs.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

RUN_SETUP="${RUN_SETUP:-0}"
RUN_TESTS="${RUN_TESTS:-0}"
LEANQUANT_CODEBOOKS="${LEANQUANT_CODEBOOKS:-leanquant}"

echo "=== RBVTQuant Colab LeanQuant-only runner ==="
echo "Repository: $ROOT_DIR"
echo "Codebooks: $LEANQUANT_CODEBOOKS"

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

CODEBOOKS="$LEANQUANT_CODEBOOKS" \
bash bash/run_colab_codebooks.sh
