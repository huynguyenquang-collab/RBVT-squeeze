#!/usr/bin/env bash
set -euo pipefail

# Resumable SqueezeLLM benchmark for a Linux GPU server:
# SqueezeLLM dense+sparse+sensitive x 3/4-bit x RTN/RBVT on Llama-3.1-8B.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT_DIR/.env"
  set +a
fi

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-server}"
if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
  elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  elif [ -x "$VENV_DIR/bin/python" ]; then
    PYTHON_BIN="$VENV_DIR/bin/python"
  else
    PYTHON_BIN="$(command -v python || command -v python3 || true)"
  fi
fi
RUN_SETUP="${RUN_SETUP:-0}"
RUN_TESTS="${RUN_TESTS:-0}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
DEVICE="${DEVICE:-cuda:0}"
BITS="${BITS:-4 3}"
METHODS="${METHODS:-rtn rbvt}"
SQUEEZELLM_MODE="${SQUEEZELLM_MODE:-hybrid}"

OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/squeezellm_server}"
STATISTICS_CACHE_DIR="${STATISTICS_CACHE_DIR:-$OUTPUT_ROOT/_statistics}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT_DIR/.cache}"
HF_HOME="${HF_HOME:-$CACHE_ROOT/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-$CACHE_ROOT/evaluation}"
LM_EVAL_OUTPUT_DIR="${LM_EVAL_OUTPUT_DIR:-$OUTPUT_ROOT/lm_eval}"

N_CALIB="${N_CALIB:-128}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
SEED="${SEED:-42}"
ROW_CHUNK="${ROW_CHUNK:-1024}"
GPTQ_BLOCKSIZE="${GPTQ_BLOCKSIZE:-128}"
KMEANS_SEED="${KMEANS_SEED:-0}"
SQUEEZELLM_SENSITIVITY="${SQUEEZELLM_SENSITIVITY:-}"
SQUEEZELLM_FISHER_SAMPLES="${SQUEEZELLM_FISHER_SAMPLES:-100}"
SQUEEZELLM_FISHER_LENGTH="${SQUEEZELLM_FISHER_LENGTH:-512}"
SQUEEZELLM_OUTLIER_RANGE="${SQUEEZELLM_OUTLIER_RANGE:-1.8}"
SQUEEZELLM_SENSITIVE_PERCENT="${SQUEEZELLM_SENSITIVE_PERCENT:-0.05}"

RBVT_LAMBDA="${RBVT_LAMBDA:-1.0}"
RBVT_TOPK="${RBVT_TOPK:-0}"
GAP_FLOOR="${GAP_FLOOR:-1e-8}"

EVAL_STRIDE="${EVAL_STRIDE:-512}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-2048}"
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"
INCLUDE_LM_EVAL="${INCLUDE_LM_EVAL:-1}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-auto}"
LM_EVAL_NUM_FEWSHOT="${LM_EVAL_NUM_FEWSHOT:-}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-}"
KEEP_MODEL="${KEEP_MODEL:-0}"
CLEAN_STATISTICS_CACHE="${CLEAN_STATISTICS_CACHE:-0}"
LM_EVAL_TASKS="${LM_EVAL_TASKS:-}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-rbvtquant}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

MIN_GPU_MEMORY_GIB="${MIN_GPU_MEMORY_GIB:-30}"
ALLOW_LOW_VRAM="${ALLOW_LOW_VRAM:-0}"

export HF_HOME
export HF_DATASETS_CACHE
export TRANSFORMERS_CACHE
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
# SqueezeLLM runs row-wise sklearn KMeans in multiprocessing workers.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1

bash bash/ensure_upstream_submodules.sh

if [ "$RUN_SETUP" = "1" ]; then
  VENV_DIR="$VENV_DIR" CACHE_ROOT="$CACHE_ROOT" \
    bash bash/setup_server_leanquant.sh
fi

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Error: Python environment not found at $PYTHON_BIN." >&2
  echo "Run: bash bash/setup_server_leanquant.sh" >&2
  exit 1
fi

if [ -z "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-${HUGGINGFACE_TOKEN:-}}}" ]; then
  echo "Error: HF_TOKEN is required for $MODEL." >&2
  exit 1
fi

if [ "$RUN_TESTS" = "1" ]; then
  PYTHON_BIN="$PYTHON_BIN" VENV_DIR="$VENV_DIR" LM_EVAL_TASKS="$LM_EVAL_TASKS" \
    bash bash/test_colab_codebooks.sh
fi

if [ "$RUN_PREFLIGHT" = "1" ]; then
  "$PYTHON_BIN" - "$MODEL" "$MIN_GPU_MEMORY_GIB" "$ALLOW_LOW_VRAM" "$SQUEEZELLM_MODE" <<'PY'
import sys

import torch
from transformers import AutoConfig

model = sys.argv[1]
required_gib = float(sys.argv[2])
allow_low_vram = sys.argv[3] == "1"
mode = sys.argv[4]

if sys.version_info[:2] != (3, 12):
    raise SystemExit(
        f"Python 3.12 is required, got {sys.version_info.major}.{sys.version_info.minor}"
    )
if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")

available_gib = torch.cuda.get_device_properties(0).total_memory / 1024**3
print("GPU:", torch.cuda.get_device_name(0))
print(f"GPU memory: {available_gib:.2f} GiB")
if available_gib < required_gib:
    message = (
        f"GPU memory is {available_gib:.2f} GiB; the default workflow expects "
        f"about {required_gib:.0f} GiB."
    )
    if allow_low_vram:
        print("Warning:", message)
    else:
        raise SystemExit(message + " Set ALLOW_LOW_VRAM=1 to bypass this check.")

config = AutoConfig.from_pretrained(model, trust_remote_code=True)
print("Model config:", config.model_type)

from quantizers.upstream_imports import (
    load_squeezellm_gradients,
    load_squeezellm_kmeans,
    load_squeezellm_model_parse,
)

get_loaders, get_modules, square_grad_hook = load_squeezellm_gradients()
kmeans_fit = load_squeezellm_kmeans()
model_parse = load_squeezellm_model_parse()
print("SqueezeLLM KMeans:", kmeans_fit.__module__)
print("SqueezeLLM model parser:", model_parse.__name__)
print("SqueezeLLM gradients loader:", get_loaders.__module__)
print("SqueezeLLM gradients modules:", get_modules.__upstream_source__)
print("SqueezeLLM gradients hook:", square_grad_hook.__upstream_source__)
if mode == "hybrid":
    from quantizers.upstream_imports import load_squeezellm_remove_outliers

    remove_outliers = load_squeezellm_remove_outliers()
    print("SqueezeLLM sparse extractor:", remove_outliers.__module__)
PY
fi

mkdir -p \
  "$OUTPUT_ROOT" \
  "$STATISTICS_CACHE_DIR" \
  "$LOG_DIR" \
  "$HF_DATASETS_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$EVAL_CACHE_DIR" \
  "$LM_EVAL_OUTPUT_DIR"

read -r -a BITS_ARRAY <<< "$BITS"
read -r -a METHOD_ARRAY <<< "$METHODS"

MODEL_SLUG="$(
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" - "$MODEL" <<'PY'
import sys
from runtime_utils import build_model_slug

print(build_model_slug(sys.argv[1]))
PY
)"

if [ "$SQUEEZELLM_MODE" = "dense-only" ]; then
  SQUEEZELLM_CACHE_VERSION="dense_only"
else
  SQUEEZELLM_CACHE_VERSION="hybrid_range${SQUEEZELLM_OUTLIER_RANGE}_sensitive${SQUEEZELLM_SENSITIVE_PERCENT}"
fi

summary_has_full_metrics() {
  local summary_path="$1"
  local require_lm_eval="$2"
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" - "$summary_path" "$require_lm_eval" <<'PY'
import json
import sys
from pathlib import Path

from runtime_utils import pick_lm_eval_metric

summary_path = Path(sys.argv[1])
require_lm_eval = sys.argv[2] == "1"
if not summary_path.exists():
    raise SystemExit(1)
summary = json.loads(summary_path.read_text(encoding="utf-8"))
perplexity = summary.get("evaluation", {}).get("perplexity", {})
for dataset in ("WikiText-2", "C4"):
    value = perplexity.get(dataset, {}).get("perplexity")
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise SystemExit(1)
if not require_lm_eval:
    raise SystemExit(0)
run_label = summary.get("run_label")
task_summary = (
    summary.get("evaluation", {})
    .get("lm_eval", {})
    .get(run_label, {})
    .get("summary", {})
)
requested_tasks = summary.get("args", {}).get("lm_eval_tasks", [])
if not isinstance(requested_tasks, list):
    raise SystemExit(1)
for task_name in requested_tasks:
    metrics = task_summary.get(task_name, {})
    if task_name == "gsm8k":
        for metric_name in ("exact_match,strict-match", "exact_match,flexible-extract"):
            value = metrics.get(metric_name)
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise SystemExit(1)
    _, score = pick_lm_eval_metric(metrics)
    if score is None:
        raise SystemExit(1)
PY
}

squeezellm_codebook_complete() {
  local bits="$1"
  local manifest="$STATISTICS_CACHE_DIR/codebooks/$MODEL_SLUG/squeezellm_${bits}bit_${SQUEEZELLM_CACHE_VERSION}/manifest.json"
  PYTHONDONTWRITEBYTECODE=1 "$PYTHON_BIN" - "$manifest" <<'PY'
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
if not manifest.exists():
    raise SystemExit(1)
data = json.loads(manifest.read_text(encoding="utf-8"))
raise SystemExit(0 if data.get("complete") else 1)
PY
}

squeezellm_bit_methods_complete() {
  local bits="$1"
  local method summary_path
  for method in "${METHOD_ARRAY[@]}"; do
    summary_path="$OUTPUT_ROOT/squeezellm_${bits}bit_${method}/run_summary.json"
    if ! summary_has_full_metrics "$summary_path" "$INCLUDE_LM_EVAL"; then
      return 1
    fi
  done
}

squeezellm_all_requested_complete() {
  local bits
  for bits in "${BITS_ARRAY[@]}"; do
    if ! squeezellm_bit_methods_complete "$bits"; then
      return 1
    fi
  done
}

cleanup_completed_squeezellm_statistics() {
  [ "$CLEAN_STATISTICS_CACHE" = "1" ] || return 0

  local bits sparse_dir fisher_dir codebook_dir
  fisher_dir="$STATISTICS_CACHE_DIR/fisher/$MODEL_SLUG/c4_n${SQUEEZELLM_FISHER_SAMPLES}_len${SQUEEZELLM_FISHER_LENGTH}_seed0_gradients_5f2a166"
  for bits in "${BITS_ARRAY[@]}"; do
    sparse_dir="$STATISTICS_CACHE_DIR/sparse_residuals/$MODEL_SLUG/squeezellm_${bits}bit_range${SQUEEZELLM_OUTLIER_RANGE}_sensitive${SQUEEZELLM_SENSITIVE_PERCENT}"
    codebook_dir="$STATISTICS_CACHE_DIR/codebooks/$MODEL_SLUG/squeezellm_${bits}bit_${SQUEEZELLM_CACHE_VERSION}"

    if squeezellm_bit_methods_complete "$bits"; then
      if [ -d "$codebook_dir" ]; then
        echo "Removing completed SqueezeLLM codebook cache: $codebook_dir"
        rm -rf "$codebook_dir"
      fi
      if [ "$SQUEEZELLM_MODE" = "hybrid" ] && [ -d "$sparse_dir" ]; then
        echo "Removing completed SqueezeLLM sparse residual cache: $sparse_dir"
        rm -rf "$sparse_dir"
      fi
      if [ -z "$SQUEEZELLM_SENSITIVITY" ] && [ -d "$fisher_dir" ]; then
        echo "Removing completed SqueezeLLM Fisher cache: $fisher_dir"
        rm -rf "$fisher_dir"
      fi
      if [ -d "$STATISTICS_CACHE_DIR/calibration" ]; then
        echo "Removing completed calibration cache: $STATISTICS_CACHE_DIR/calibration"
        rm -rf "$STATISTICS_CACHE_DIR/calibration"
      fi
    fi
  done

  if squeezellm_all_requested_complete; then
    find "$STATISTICS_CACHE_DIR" -type d -empty -delete 2>/dev/null || true
  fi
}

COMMON_ARGS=(
  --model-path "$MODEL"
  --device "$DEVICE"
  --output-root "$OUTPUT_ROOT"
  --resume
  --calib-dataset c4
  --n-calib "$N_CALIB"
  --max-length "$MAX_LENGTH"
  --seed "$SEED"
  --group-size -1
  --row-chunk "$ROW_CHUNK"
  --gptq-blocksize "$GPTQ_BLOCKSIZE"
  --kmeans-seed "$KMEANS_SEED"
  --squeezellm-fisher-samples "$SQUEEZELLM_FISHER_SAMPLES"
  --squeezellm-fisher-length "$SQUEEZELLM_FISHER_LENGTH"
  --squeezellm-mode "$SQUEEZELLM_MODE"
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
if [ "$INCLUDE_LM_EVAL" = "1" ]; then
  COMMON_ARGS+=(--include-lm-eval)
else
  COMMON_ARGS+=(--no-lm-eval)
fi
if [ -n "$LM_EVAL_NUM_FEWSHOT" ]; then
  COMMON_ARGS+=(--lm-eval-num-fewshot "$LM_EVAL_NUM_FEWSHOT")
fi
if [ -n "$LM_EVAL_LIMIT" ]; then
  COMMON_ARGS+=(--lm-eval-limit "$LM_EVAL_LIMIT")
fi
if [ -n "$LM_EVAL_TASKS" ]; then
  read -r -a LM_EVAL_TASK_ARRAY <<< "$LM_EVAL_TASKS"
  COMMON_ARGS+=(--lm-eval-tasks "${LM_EVAL_TASK_ARRAY[@]}")
fi
if [ "$KEEP_MODEL" = "1" ]; then
  COMMON_ARGS+=(--keep-model)
fi
if [ "$USE_WANDB" = "1" ]; then
  COMMON_ARGS+=(--use-wandb --wandb-project "$WANDB_PROJECT")
  if [ -n "$WANDB_ENTITY" ]; then
    COMMON_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
else
  COMMON_ARGS+=(--no-wandb)
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/squeezellm_${TIMESTAMP}.log"

{
  echo "=== RBVTQuant SqueezeLLM server benchmark ==="
  echo "Repository: $ROOT_DIR"
  echo "Model: $MODEL"
  echo "Device: $DEVICE"
  echo "Bits: $BITS"
  echo "Methods: $METHODS"
  echo "SqueezeLLM mode: $SQUEEZELLM_MODE"
  echo "SqueezeLLM Fisher: C4/${SQUEEZELLM_FISHER_SAMPLES}x${SQUEEZELLM_FISHER_LENGTH}"
  echo "SqueezeLLM outlier range: $SQUEEZELLM_OUTLIER_RANGE"
  echo "SqueezeLLM sensitive values: $SQUEEZELLM_SENSITIVE_PERCENT%"
  echo "RBVT lambda: $RBVT_LAMBDA"
  echo "Worker threads: OMP=$OMP_NUM_THREADS | MKL=$MKL_NUM_THREADS"
  echo "RBVT calibration: C4/${N_CALIB}x${MAX_LENGTH}"
  echo "LM-eval tasks override: ${LM_EVAL_TASKS:-default}"
  echo "Output: $OUTPUT_ROOT"
  echo "Statistics cache: $STATISTICS_CACHE_DIR"
  echo "Clean non-codebook statistics cache after completed results: $CLEAN_STATISTICS_CACHE"
  echo "W&B logging: $USE_WANDB | project=$WANDB_PROJECT | entity=${WANDB_ENTITY:-default}"
  nvidia-smi
} 2>&1 | tee -a "$LOG_FILE"

cleanup_completed_squeezellm_statistics

run_index=0
total_runs=$((${#BITS_ARRAY[@]} * ${#METHOD_ARRAY[@]}))
for bits in "${BITS_ARRAY[@]}"; do
  for method in "${METHOD_ARRAY[@]}"; do
    run_index=$((run_index + 1))
    {
      echo
      echo "=== Job ${run_index}/${total_runs}: SqueezeLLM ${bits}-bit ${method} ==="
      "$PYTHON_BIN" codebook_benchmark.py \
        "${COMMON_ARGS[@]}" \
        --codebooks squeezellm \
        --bits "$bits" \
        --methods "$method"
    } 2>&1 | tee -a "$LOG_FILE"
    cleanup_completed_squeezellm_statistics
  done
done

{
  echo
  echo "=== Building combined SqueezeLLM report ==="
  "$PYTHON_BIN" codebook_benchmark.py \
    "${COMMON_ARGS[@]}" \
    --codebooks squeezellm \
    --bits "${BITS_ARRAY[@]}" \
    --methods "${METHOD_ARRAY[@]}"
} 2>&1 | tee -a "$LOG_FILE"

echo "Benchmark complete."
echo "Results: $OUTPUT_ROOT/benchmark_results.csv"
echo "Markdown: $OUTPUT_ROOT/benchmark_results.md"
echo "Log: $LOG_FILE"
