#!/usr/bin/env bash
set -euo pipefail

# Report-only SqueezeLLM 3-bit RBVT sweep for lambda 0.9 and 1.5.
# It skips RTN and reuses the existing SqueezeLLM Fisher/codebook cache.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

LAMBDA_VALUES="${LAMBDA_VALUES:-0.9 1.5}" \
RUN_RTN=0 \
SELECT_BEST=0 \
LOCAL_OUTPUT_ROOT="${LOCAL_OUTPUT_ROOT:-/content/rbvtquant_outputs/squeezellm_lambda_0p9_1p5}" \
DRIVE_OUTPUT_ROOT="${DRIVE_OUTPUT_ROOT:-/content/drive/MyDrive/RBVTQuant/squeezellm_lambda_0p9_1p5}" \
STATISTICS_CACHE_DIR="${STATISTICS_CACHE_DIR:-/content/rbvtquant_outputs/squeezellm_lambda_sweep/_statistics}" \
bash bash/run_colab_squeezellm_lambda_sweep.sh
