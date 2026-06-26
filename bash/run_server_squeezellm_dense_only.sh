#!/usr/bin/env bash
set -euo pipefail

# SqueezeLLM upstream dense-only x 3/4-bit x RTN/RBVT on a Linux GPU server.
# Outputs and all statistics caches are isolated from the hybrid workflow.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export SQUEEZELLM_MODE="dense-only"
export BITS="${BITS:-4 3}"
export METHODS="${METHODS:-rtn rbvt}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/squeezellm_dense_only_server}"
export STATISTICS_CACHE_DIR="${STATISTICS_CACHE_DIR:-$OUTPUT_ROOT/_statistics}"
export LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"
export LM_EVAL_OUTPUT_DIR="${LM_EVAL_OUTPUT_DIR:-$OUTPUT_ROOT/lm_eval}"

exec bash "$ROOT_DIR/bash/run_server_squeezellm.sh"
