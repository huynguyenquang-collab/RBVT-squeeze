#!/usr/bin/env bash
set -euo pipefail

# Resumable Google Colab benchmark:
# LeanQuant/SqueezeLLM x 3/4-bit x RTN/RBVT on Llama-3.1-8B.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-colab}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
DEVICE="${DEVICE:-cuda:0}"
CODEBOOKS="${CODEBOOKS:-leanquant squeezellm}"
BITS="${BITS:-4 3}"
METHODS="${METHODS:-rtn rbvt}"

LOCAL_OUTPUT_ROOT="${LOCAL_OUTPUT_ROOT:-/content/rbvtquant_outputs/codebook_benchmark}"
DRIVE_OUTPUT_ROOT="${DRIVE_OUTPUT_ROOT:-/content/drive/MyDrive/RBVTQuant/codebook_benchmark}"
USE_DRIVE="${USE_DRIVE:-auto}"
LOG_DIR="${LOG_DIR:-}"

CALIB_DATASET="${CALIB_DATASET:-c4}"
N_CALIB="${N_CALIB:-128}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
SEED="${SEED:-42}"

GROUP_SIZE="${GROUP_SIZE:--1}"
ROW_CHUNK="${ROW_CHUNK:-1024}"
LEANQUANT_EXPONENT="${LEANQUANT_EXPONENT:-4.0}"
LEANQUANT_PERCDAMP="${LEANQUANT_PERCDAMP:-0.1}"
KMEANS_SEED="${KMEANS_SEED:-0}"
SQUEEZELLM_SENSITIVITY="${SQUEEZELLM_SENSITIVITY:-}"
SQUEEZELLM_FISHER_SAMPLES="${SQUEEZELLM_FISHER_SAMPLES:-100}"
SQUEEZELLM_FISHER_LENGTH="${SQUEEZELLM_FISHER_LENGTH:-512}"
SQUEEZELLM_OUTLIER_RANGE="${SQUEEZELLM_OUTLIER_RANGE:-1.8}"
SQUEEZELLM_SENSITIVE_PERCENT="${SQUEEZELLM_SENSITIVE_PERCENT:-0.05}"
STATISTICS_CACHE_DIR="${STATISTICS_CACHE_DIR:-$LOCAL_OUTPUT_ROOT/_statistics}"

RBVT_LAMBDA="${RBVT_LAMBDA:-1.0}"
RBVT_TOPK="${RBVT_TOPK:-0}"
GAP_FLOOR="${GAP_FLOOR:-1e-8}"

EVAL_STRIDE="${EVAL_STRIDE:-512}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-2048}"
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-auto}"
LM_EVAL_NUM_FEWSHOT="${LM_EVAL_NUM_FEWSHOT:-}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-}"
INCLUDE_LM_EVAL="${INCLUDE_LM_EVAL:-1}"
KEEP_MODEL="${KEEP_MODEL:-0}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"

HF_HOME="${HF_HOME:-/content/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/content/huggingface/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/content/huggingface/transformers}"
CALIB_CACHE_DIR="$ROOT_DIR/calibration_cache"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-$ROOT_DIR/dataset_cache}"
LM_EVAL_OUTPUT_DIR="${LM_EVAL_OUTPUT_DIR:-$LOCAL_OUTPUT_ROOT/lm_eval}"

export HF_HOME
export HF_DATASETS_CACHE
export TRANSFORMERS_CACHE
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Error: Python environment not found at $PYTHON_BIN." >&2
  echo "Run: bash bash/setup_colab_codebooks.sh" >&2
  exit 1
fi

if [ -z "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-${HUGGINGFACE_TOKEN:-}}}" ]; then
  echo "Error: HF_TOKEN is required for $MODEL." >&2
  exit 1
fi

if [ "$USE_DRIVE" = "auto" ]; then
  if [ -d /content/drive/MyDrive ]; then
    USE_DRIVE=1
  else
    USE_DRIVE=0
  fi
fi
if [ "$USE_DRIVE" != "0" ] && [ "$USE_DRIVE" != "1" ]; then
  echo "Error: USE_DRIVE must be auto, 0, or 1; got $USE_DRIVE." >&2
  exit 1
fi

if [ "$USE_DRIVE" = "1" ] && [ ! -d /content/drive/MyDrive ]; then
  echo "Error: Google Drive is not mounted at /content/drive/MyDrive." >&2
  echo "Mount Drive in a Colab cell or run with USE_DRIVE=0." >&2
  exit 1
fi

if [ "$RUN_PREFLIGHT" = "1" ]; then
  MODEL="$MODEL" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash bash/check_colab_codebooks.sh
fi

if [ -z "$LOG_DIR" ]; then
  if [ "$USE_DRIVE" = "1" ]; then
    LOG_DIR="$DRIVE_OUTPUT_ROOT/logs"
  else
    LOG_DIR="$LOCAL_OUTPUT_ROOT/logs"
  fi
fi

mkdir -p \
  "$LOCAL_OUTPUT_ROOT" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$CALIB_CACHE_DIR" \
  "$EVAL_CACHE_DIR" \
  "$LM_EVAL_OUTPUT_DIR"

if [ "$USE_DRIVE" = "1" ]; then
  mkdir -p "$DRIVE_OUTPUT_ROOT" "$LOG_DIR"
  if [ -d "$DRIVE_OUTPUT_ROOT" ]; then
    cp -a "$DRIVE_OUTPUT_ROOT/." "$LOCAL_OUTPUT_ROOT/"
  fi
else
  mkdir -p "$LOG_DIR"
fi

read -r -a CODEBOOK_ARRAY <<< "$CODEBOOKS"
read -r -a BITS_ARRAY <<< "$BITS"
read -r -a METHOD_ARRAY <<< "$METHODS"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/codebook_benchmark_${TIMESTAMP}.log"

sync_results() {
  if [ "$USE_DRIVE" = "1" ]; then
    mkdir -p "$DRIVE_OUTPUT_ROOT"
    cp -a "$LOCAL_OUTPUT_ROOT/." "$DRIVE_OUTPUT_ROOT/"
  fi
}

build_common_args() {
  COMMON_ARGS=(
    --model-path "$MODEL"
    --device "$DEVICE"
    --output-root "$LOCAL_OUTPUT_ROOT"
    --resume
    --calib-dataset "$CALIB_DATASET"
    --n-calib "$N_CALIB"
    --max-length "$MAX_LENGTH"
    --seed "$SEED"
    --group-size "$GROUP_SIZE"
    --row-chunk "$ROW_CHUNK"
    --leanquant-exponent "$LEANQUANT_EXPONENT"
    --leanquant-percdamp "$LEANQUANT_PERCDAMP"
    --kmeans-seed "$KMEANS_SEED"
    --squeezellm-fisher-samples "$SQUEEZELLM_FISHER_SAMPLES"
    --squeezellm-fisher-length "$SQUEEZELLM_FISHER_LENGTH"
    --squeezellm-outlier-range "$SQUEEZELLM_OUTLIER_RANGE"
    --squeezellm-sensitive-percent "$SQUEEZELLM_SENSITIVE_PERCENT"
    --statistics-cache-dir "$STATISTICS_CACHE_DIR"
    --rbvt-lambda "$RBVT_LAMBDA"
    --rbvt-topk "$RBVT_TOPK"
    --gap-floor "$GAP_FLOOR"
    --eval-stride "$EVAL_STRIDE"
    --eval-max-length "$EVAL_MAX_LENGTH"
    --eval-samples "$EVAL_SAMPLES"
    --eval-cache-dir "$EVAL_CACHE_DIR"
    --lm-eval-batch-size "$LM_EVAL_BATCH_SIZE"
    --lm-eval-output-dir "$LM_EVAL_OUTPUT_DIR"
  )

  if [ -n "$SQUEEZELLM_SENSITIVITY" ]; then
    COMMON_ARGS+=(--squeezellm-sensitivity "$SQUEEZELLM_SENSITIVITY")
  fi
  if [ -n "$LM_EVAL_NUM_FEWSHOT" ]; then
    COMMON_ARGS+=(--lm-eval-num-fewshot "$LM_EVAL_NUM_FEWSHOT")
  fi
  if [ -n "$LM_EVAL_LIMIT" ]; then
    COMMON_ARGS+=(--lm-eval-limit "$LM_EVAL_LIMIT")
  fi
  if [ "$INCLUDE_LM_EVAL" = "1" ]; then
    COMMON_ARGS+=(--include-lm-eval)
  else
    COMMON_ARGS+=(--no-lm-eval)
  fi
  if [ "$KEEP_MODEL" = "1" ]; then
    COMMON_ARGS+=(--keep-model)
  fi
}

trap sync_results EXIT
build_common_args

{
  echo "=== RBVTQuant Colab codebook benchmark ==="
  echo "Model: $MODEL"
  echo "Device: $DEVICE"
  echo "Codebooks: $CODEBOOKS"
  echo "Bits: $BITS"
  echo "Methods: $METHODS"
  echo "RBVT row chunk: $ROW_CHUNK"
  echo "LeanQuant: exponent=$LEANQUANT_EXPONENT | percdamp=$LEANQUANT_PERCDAMP | act_order=1"
  echo "SqueezeLLM Fisher: samples=$SQUEEZELLM_FISHER_SAMPLES | length=$SQUEEZELLM_FISHER_LENGTH | external=${SQUEEZELLM_SENSITIVITY:-auto-collect}"
  echo "SqueezeLLM mode: dense+sparse+sensitive | outlier_range=$SQUEEZELLM_OUTLIER_RANGE | sensitive=$SQUEEZELLM_SENSITIVE_PERCENT%"
  echo "Statistics cache: $STATISTICS_CACHE_DIR"
  echo "Local output: $LOCAL_OUTPUT_ROOT"
  echo "Drive output: $DRIVE_OUTPUT_ROOT"
  echo "Drive sync enabled: $USE_DRIVE"
  echo "HF cache: $HF_HOME"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
  else
    echo "Warning: nvidia-smi is unavailable; preflight should be enabled for real runs."
  fi
} 2>&1 | tee -a "$LOG_FILE"

run_index=0
total_runs=$((${#CODEBOOK_ARRAY[@]} * ${#BITS_ARRAY[@]} * ${#METHOD_ARRAY[@]}))

for codebook in "${CODEBOOK_ARRAY[@]}"; do
  for bits in "${BITS_ARRAY[@]}"; do
    for method in "${METHOD_ARRAY[@]}"; do
      run_index=$((run_index + 1))
      {
        echo
        echo "=== Job ${run_index}/${total_runs}: ${codebook} ${bits}-bit ${method} ==="
        "$PYTHON_BIN" codebook_benchmark.py \
          "${COMMON_ARGS[@]}" \
          --codebooks "$codebook" \
          --bits "$bits" \
          --methods "$method"
      } 2>&1 | tee -a "$LOG_FILE"
      sync_results
    done
  done
done

# All summaries now exist locally. A final resume-only pass generates one
# combined JSON/CSV/Markdown table containing the complete requested matrix.
{
  echo
  echo "=== Building combined benchmark report ==="
  "$PYTHON_BIN" codebook_benchmark.py \
    "${COMMON_ARGS[@]}" \
    --codebooks "${CODEBOOK_ARRAY[@]}" \
    --bits "${BITS_ARRAY[@]}" \
    --methods "${METHOD_ARRAY[@]}"
} 2>&1 | tee -a "$LOG_FILE"

sync_results

echo "Benchmark complete."
echo "Results: $LOCAL_OUTPUT_ROOT/benchmark_results.csv"
if [ "$USE_DRIVE" = "1" ]; then
  echo "Persistent results: $DRIVE_OUTPUT_ROOT/benchmark_results.csv"
fi
echo "Log: $LOG_FILE"
