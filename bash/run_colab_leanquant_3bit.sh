#!/usr/bin/env bash
set -euo pipefail

# Colab entrypoint for LeanQuant 3-bit RTN and RBVT only.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

CODEBOOKS=leanquant \
BITS=3 \
METHODS="rtn rbvt" \
LOCAL_OUTPUT_ROOT="${LOCAL_OUTPUT_ROOT:-/content/rbvtquant_outputs/leanquant_3bit}" \
DRIVE_OUTPUT_ROOT="${DRIVE_OUTPUT_ROOT:-/content/drive/MyDrive/RBVTQuant/leanquant_3bit}" \
STATISTICS_CACHE_DIR="${STATISTICS_CACHE_DIR:-/content/rbvtquant_outputs/codebook_benchmark/_statistics}" \
bash bash/run_colab_leanquant.sh
