#!/usr/bin/env bash
set -euo pipefail

# Preflight checks before starting the Llama-3.1-8B codebook benchmark.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-colab}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
MIN_GPU_MEMORY_GIB="${MIN_GPU_MEMORY_GIB:-30}"
MIN_DISK_GIB="${MIN_DISK_GIB:-40}"
ALLOW_LOW_VRAM="${ALLOW_LOW_VRAM:-0}"
ALLOW_LOW_DISK="${ALLOW_LOW_DISK:-0}"
CHECK_MODEL_ACCESS="${CHECK_MODEL_ACCESS:-1}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Error: Python environment not found at $PYTHON_BIN." >&2
  echo "Run: bash bash/setup_colab_codebooks.sh" >&2
  exit 1
fi

echo "=== RBVTQuant codebook preflight ==="
echo "Repository: $ROOT_DIR"
echo "Python: $PYTHON_BIN"
echo "Model: $MODEL"

"$PYTHON_BIN" - <<'PY'
import importlib
import sys

required = [
    "torch",
    "transformers",
    "datasets",
    "accelerate",
    "safetensors",
    "sklearn",
    "lm_eval",
]

print("Python version:", sys.version.replace("\n", " "))
if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Python 3.12 is required, got {sys.version_info.major}.{sys.version_info.minor}")

for module_name in required:
    module = importlib.import_module(module_name)
    print(f"{module_name}: {getattr(module, '__version__', 'installed')}")
PY

GPU_MEMORY_GIB="$("$PYTHON_BIN" - <<'PY'
import torch

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable")

properties = torch.cuda.get_device_properties(0)
print(f"{properties.total_memory / 1024**3:.2f}")
PY
)"

echo "GPU memory: ${GPU_MEMORY_GIB} GiB"
"$PYTHON_BIN" - "$GPU_MEMORY_GIB" "$MIN_GPU_MEMORY_GIB" "$ALLOW_LOW_VRAM" <<'PY'
import sys

available = float(sys.argv[1])
required = float(sys.argv[2])
allow_low_vram = sys.argv[3] == "1"
if available < required:
    message = (
        f"GPU memory is {available:.2f} GiB; the default full Llama-3.1-8B "
        f"workflow expects about {required:.0f} GiB or more."
    )
    if allow_low_vram:
        print("Warning:", message)
    else:
        raise SystemExit(message + " Set ALLOW_LOW_VRAM=1 to bypass this check.")
PY

DISK_PATH="/content"
if [ ! -d "$DISK_PATH" ]; then
  DISK_PATH="$ROOT_DIR"
fi
DISK_AVAILABLE_GIB="$(df -Pk "$DISK_PATH" | awk 'NR == 2 {printf "%.2f", $4 / 1024 / 1024}')"
echo "Free disk at $DISK_PATH: ${DISK_AVAILABLE_GIB} GiB"
"$PYTHON_BIN" - "$DISK_AVAILABLE_GIB" "$MIN_DISK_GIB" "$ALLOW_LOW_DISK" <<'PY'
import sys

available = float(sys.argv[1])
required = float(sys.argv[2])
allow_low_disk = sys.argv[3] == "1"
if available < required:
    message = (
        f"Free disk is {available:.2f} GiB; model cache and temporary checkpoints "
        f"need about {required:.0f} GiB or more."
    )
    if allow_low_disk:
        print("Warning:", message)
    else:
        raise SystemExit(message + " Set ALLOW_LOW_DISK=1 to bypass this check.")
PY

if [ -z "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-${HUGGINGFACE_TOKEN:-}}}" ]; then
  echo "Error: HF_TOKEN is missing. Llama-3.1-8B is gated." >&2
  echo "Accept the Meta Llama license and export HF_TOKEN before running." >&2
  exit 1
fi

if [ "$CHECK_MODEL_ACCESS" = "1" ]; then
  echo "Checking Hugging Face access to $MODEL ..."
  MODEL="$MODEL" "$PYTHON_BIN" - <<'PY'
import os

from transformers import AutoConfig

token = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGINGFACE_HUB_TOKEN")
    or os.getenv("HUGGINGFACE_TOKEN")
)
config = AutoConfig.from_pretrained(
    os.environ["MODEL"],
    token=token,
    trust_remote_code=True,
)
print("Model config:", config.model_type)
PY
fi

"$PYTHON_BIN" -m py_compile \
  codebook_benchmark.py \
  lm_eval_dataset_smoke.py \
  lm_eval_runner.py \
  quantizers/base_codebook.py \
  quantizers/codebook_store.py \
  quantizers/hessian_store.py \
  quantizers/codebook_factory.py \
  quantizers/leanquant_collector.py \
  quantizers/sensitivity_store.py \
  quantizers/sparse_residual_store.py \
  quantizers/squeezellm_collector.py \
  quantizers/upstream_imports.py \
  quantizers/upstream_calibration.py \
  LeanQuant/lean_quantizer.py \
  SqueezeLLM-gradients/datautils.py \
  SqueezeLLM-gradients/run.py \
  SqueezeLLM/quantization/nuq.py \
  SqueezeLLM/squeezellm/model_parse.py

"$PYTHON_BIN" - <<'PY'
import sys

from quantizers.upstream_imports import (
    load_leanquant_upstream,
    load_squeezellm_gradients,
    load_squeezellm_kmeans,
    load_squeezellm_model_parse,
)

LeanQuant, Quantizer = load_leanquant_upstream()
get_loaders, get_modules, square_grad_hook = load_squeezellm_gradients()
kmeans_fit = load_squeezellm_kmeans()
model_parse = load_squeezellm_model_parse()
print("LeanQuant upstream:", LeanQuant.__module__, Quantizer.__module__)
print("SqueezeLLM upstream:", kmeans_fit.__module__, model_parse.__name__)
print(
    "SqueezeLLM-gradients upstream:",
    get_loaders.__module__,
    get_modules.__upstream_source__,
    square_grad_hook.__upstream_source__,
)

leanquant_module = sys.modules.get("lean_quantizer")
pool = getattr(leanquant_module, "pool", None)
if pool is not None:
    pool.close()
    pool.join()
PY

echo "Preflight passed."
