#!/usr/bin/env bash
set -euo pipefail

# Qwen2.5 SqueezeLLM dense-only RTN/RBVT with RedPajama calibration.
# Defaults are the paper-style RedPajama calibration config: 1024 samples x 4096 tokens.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT_DIR/.env"
  set +a
fi

if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -n "${VIRTUAL_ENV:-}" ] && [ -x "${VIRTUAL_ENV}/bin/python" ]; then
    PYTHON_BIN="${VIRTUAL_ENV}/bin/python"
  elif [ -n "${CONDA_PREFIX:-}" ] && [ -x "${CONDA_PREFIX}/bin/python" ]; then
    PYTHON_BIN="${CONDA_PREFIX}/bin/python"
  else
    PYTHON_BIN="$(command -v python || command -v python3 || true)"
  fi
fi

MODEL="${MODEL:-Qwen/Qwen2.5-7B}"
DEVICE="${DEVICE:-cuda:0}"
BIT="${BIT:-3}"
METHODS="${METHODS:-rtn rbvt}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$ROOT_DIR/outputs/qwen25_squeezellm_redpajama}"
STATISTICS_CACHE_DIR="${STATISTICS_CACHE_DIR:-$OUTPUT_ROOT/_statistics}"
LOG_DIR="${LOG_DIR:-$OUTPUT_ROOT/logs}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-$OUTPUT_ROOT/eval_cache}"
LM_EVAL_OUTPUT_DIR="${LM_EVAL_OUTPUT_DIR:-$OUTPUT_ROOT/lm_eval}"

N_CALIB="${N_CALIB:-1024}"
MAX_LENGTH="${MAX_LENGTH:-4096}"
SEED="${SEED:-0}"
SQUEEZELLM_FISHER_DATASET="${SQUEEZELLM_FISHER_DATASET:-redpajama}"
SQUEEZELLM_FISHER_SAMPLES="${SQUEEZELLM_FISHER_SAMPLES:-1024}"
SQUEEZELLM_FISHER_LENGTH="${SQUEEZELLM_FISHER_LENGTH:-4096}"
SQUEEZELLM_FISHER_ACCUM_DEVICE="${SQUEEZELLM_FISHER_ACCUM_DEVICE:-gpu}"
SQUEEZELLM_FISHER_GRAD_CHECKPOINTING="${SQUEEZELLM_FISHER_GRAD_CHECKPOINTING:-0}"
SQUEEZELLM_FISHER_LAYERS_PER_PASS="${SQUEEZELLM_FISHER_LAYERS_PER_PASS:-1}"
SQUEEZELLM_MODE="${SQUEEZELLM_MODE:-dense-only}"

RBVT_LAMBDA="${RBVT_LAMBDA:-1.0}"
RBVT_TOPK="${RBVT_TOPK:-0}"
GAP_FLOOR="${GAP_FLOOR:-1e-8}"
ROW_CHUNK="${ROW_CHUNK:-1024}"

EVAL_STRIDE="${EVAL_STRIDE:-512}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-2048}"
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"
INCLUDE_LM_EVAL="${INCLUDE_LM_EVAL:-0}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-auto}"
LM_EVAL_TASKS="${LM_EVAL_TASKS:-arc_challenge arc_easy boolq hellaswag lambada_openai openbookqa piqa rte winogrande}"
KEEP_MODEL="${KEEP_MODEL:-0}"
USE_WANDB="${USE_WANDB:-0}"
WANDB_PROJECT="${WANDB_PROJECT:-RBVTsqueeze}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

REDPAJAMA_DATASET="${REDPAJAMA_DATASET:-ZengXiangyu/RedPajama-Data-1T-Sample}"
REDPAJAMA_STREAMING="${REDPAJAMA_STREAMING:-0}"
export REDPAJAMA_DATASET REDPAJAMA_STREAMING

# If the caller asks for one physical GPU (for example DEVICE=cuda:1), prefer
# that GPU as logical cuda:0. By default keep the other GPU visible as spill
# capacity for device_map=auto because Qwen 7B + 4096-token Fisher/RBVT can
# exceed one 40GB A100 when other processes are alive.
REQUESTED_DEVICE="$DEVICE"
GPU_SPILL="${GPU_SPILL:-0}"
MODEL_DEVICE_MAP="${MODEL_DEVICE_MAP:-}"
MODEL_MAX_MEMORY="${MODEL_MAX_MEMORY:-}"
MODEL_OFFLOAD_FOLDER="${MODEL_OFFLOAD_FOLDER:-$OUTPUT_ROOT/offload}"
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [[ "$DEVICE" =~ ^cuda:([0-9]+)$ ]]; then
  requested_gpu="${BASH_REMATCH[1]}"
  if [ "$GPU_SPILL" = "1" ] && [ "${CUDA_DEVICE_COUNT:-2}" -gt 1 ]; then
    if [ "$requested_gpu" = "0" ]; then
      export CUDA_VISIBLE_DEVICES="0,1"
    else
      export CUDA_VISIBLE_DEVICES="$requested_gpu,0"
    fi
    MODEL_DEVICE_MAP="${MODEL_DEVICE_MAP:-auto}"
    MODEL_MAX_MEMORY="${MODEL_MAX_MEMORY:-0:20GiB,1:20GiB,cpu:120GiB}"
  else
    export CUDA_VISIBLE_DEVICES="$requested_gpu"
    MODEL_DEVICE_MAP="${MODEL_DEVICE_MAP:-}"
  fi
  DEVICE="cuda:0"
fi
export MODEL_DEVICE_MAP MODEL_MAX_MEMORY MODEL_OFFLOAD_FOLDER

export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export SQUEEZELLM_FISHER_ACCUM_DEVICE SQUEEZELLM_FISHER_GRAD_CHECKPOINTING SQUEEZELLM_FISHER_LAYERS_PER_PASS
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"

mkdir -p "$OUTPUT_ROOT" "$LOG_DIR"
mkdir -p "$MODEL_OFFLOAD_FOLDER"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/qwen25_squeezellm_redpajama_${TIMESTAMP}.log"

LM_EVAL_ARGS=(--no-lm-eval)
if [ "$INCLUDE_LM_EVAL" = "1" ]; then
  read -r -a LM_EVAL_TASK_ARRAY <<< "$LM_EVAL_TASKS"
  LM_EVAL_ARGS=(
    --include-lm-eval
    --lm-eval-batch-size "$LM_EVAL_BATCH_SIZE"
    --lm-eval-output-dir "$LM_EVAL_OUTPUT_DIR"
    --lm-eval-tasks "${LM_EVAL_TASK_ARRAY[@]}"
  )
fi

WANDB_ARGS=(--no-wandb)
if [ "$USE_WANDB" = "1" ]; then
  WANDB_ARGS=(--use-wandb --wandb-project "$WANDB_PROJECT")
  if [ -n "$WANDB_ENTITY" ]; then
    WANDB_ARGS+=(--wandb-entity "$WANDB_ENTITY")
  fi
fi

KEEP_ARGS=()
if [ "$KEEP_MODEL" = "1" ]; then
  KEEP_ARGS+=(--keep-model)
fi

MODEL_PLACEMENT_ARGS=()
if [ -n "$MODEL_DEVICE_MAP" ]; then
  MODEL_PLACEMENT_ARGS+=(--model-device-map "$MODEL_DEVICE_MAP")
fi
if [ -n "$MODEL_MAX_MEMORY" ]; then
  MODEL_PLACEMENT_ARGS+=(--model-max-memory "$MODEL_MAX_MEMORY")
fi
MODEL_PLACEMENT_ARGS+=(--model-offload-folder "$MODEL_OFFLOAD_FOLDER")

{
  echo "=== Qwen2.5 SqueezeLLM RedPajama benchmark ==="
  echo "Model: $MODEL"
  echo "Requested device: $REQUESTED_DEVICE"
  echo "Runtime device: $DEVICE"
  echo "CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-<unset>}"
  echo "GPU spill: $GPU_SPILL"
  echo "Model device_map: ${MODEL_DEVICE_MAP:-<single-device>}"
  echo "Model max_memory: ${MODEL_MAX_MEMORY:-<unset>}"
  echo "Model offload folder: $MODEL_OFFLOAD_FOLDER"
  echo "Bits: $BIT"
  echo "Methods: $METHODS"
  echo "RBVT calibration: redpajama/${N_CALIB}x${MAX_LENGTH}, seed=$SEED"
  echo "SqueezeLLM Fisher: ${SQUEEZELLM_FISHER_DATASET}/${SQUEEZELLM_FISHER_SAMPLES}x${SQUEEZELLM_FISHER_LENGTH}, seed=0"
  echo "SqueezeLLM Fisher accum: $SQUEEZELLM_FISHER_ACCUM_DEVICE, grad_checkpointing=$SQUEEZELLM_FISHER_GRAD_CHECKPOINTING, layers_per_pass=$SQUEEZELLM_FISHER_LAYERS_PER_PASS"
  echo "RedPajama dataset: $REDPAJAMA_DATASET"
  echo "Evaluation: stride=$EVAL_STRIDE max_length=$EVAL_MAX_LENGTH samples=$EVAL_SAMPLES lm_eval=$INCLUDE_LM_EVAL"
  echo "Output: $OUTPUT_ROOT"

  "$PYTHON_BIN" codebook_benchmark.py \
    --model-path "$MODEL" \
    --device "$DEVICE" \
    "${MODEL_PLACEMENT_ARGS[@]}" \
    --output-root "$OUTPUT_ROOT" \
    --statistics-cache-dir "$STATISTICS_CACHE_DIR" \
    --codebooks squeezellm \
    --bits "$BIT" \
    --methods $METHODS \
    --squeezellm-mode "$SQUEEZELLM_MODE" \
    --squeezellm-fisher-dataset "$SQUEEZELLM_FISHER_DATASET" \
    --squeezellm-fisher-samples "$SQUEEZELLM_FISHER_SAMPLES" \
    --squeezellm-fisher-length "$SQUEEZELLM_FISHER_LENGTH" \
    --calib-dataset redpajama \
    --n-calib "$N_CALIB" \
    --max-length "$MAX_LENGTH" \
    --seed "$SEED" \
    --row-chunk "$ROW_CHUNK" \
    --rbvt-lambda "$RBVT_LAMBDA" \
    --rbvt-topk "$RBVT_TOPK" \
    --gap-floor "$GAP_FLOOR" \
    --eval-stride "$EVAL_STRIDE" \
    --eval-max-length "$EVAL_MAX_LENGTH" \
    --eval-samples "$EVAL_SAMPLES" \
    --eval-cache-dir "$EVAL_CACHE_DIR" \
    --resume \
    "${KEEP_ARGS[@]}" \
    "${LM_EVAL_ARGS[@]}" \
    "${WANDB_ARGS[@]}"
} 2>&1 | tee -a "$LOG_FILE"
