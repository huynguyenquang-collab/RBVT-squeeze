#!/usr/bin/env bash
set -euo pipefail

# Install the Python 3.12 environment used by the LeanQuant server runner.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-server}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
CACHE_ROOT="${CACHE_ROOT:-$ROOT_DIR/.cache}"
UV_CACHE_DIR="${UV_CACHE_DIR:-$CACHE_ROOT/uv}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-$CACHE_ROOT/pip}"

export UV_CACHE_DIR
export PIP_CACHE_DIR
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1
export BLIS_NUM_THREADS=1
export LOKY_MAX_CPU_COUNT=1

echo "=== RBVTQuant server setup ==="
echo "Repository: $ROOT_DIR"
echo "Repository commit: $(git rev-parse --short HEAD 2>/dev/null || echo unknown)"
echo "Python: $PYTHON_VERSION"
echo "Virtual environment: $VENV_DIR"

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Error: nvidia-smi is unavailable. A Linux server with an NVIDIA GPU is required." >&2
  exit 1
fi
nvidia-smi

if [ -f .gitmodules ]; then
  echo "Initializing upstream submodules ..."
  bash bash/ensure_upstream_submodules.sh
fi

mkdir -p "$UV_CACHE_DIR" "$PIP_CACHE_DIR"

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
else
  echo "Installing uv ..."
  python3 -m pip install --user --upgrade uv
  UV_BIN="$(python3 -m site --user-base)/bin/uv"
fi

if [ ! -x "$UV_BIN" ]; then
  echo "Error: uv was not found after installation." >&2
  exit 1
fi

"$UV_BIN" python install "$PYTHON_VERSION"
if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$UV_BIN" venv --python "$PYTHON_VERSION" "$VENV_DIR"
fi

echo "Installing requirements-server.txt ..."
"$UV_BIN" pip install \
  --python "$VENV_DIR/bin/python" \
  --upgrade \
  -r requirements-server.txt

"$VENV_DIR/bin/python" - <<'PY'
import sys

import numpy as np
import torch

print("Python:", sys.version.replace("\n", " "))
print("NumPy:", np.__version__)
print("PyTorch:", torch.__version__)
print("CUDA runtime:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if sys.version_info[:2] != (3, 12):
    raise SystemExit("Python 3.12 is required")
if not torch.cuda.is_available():
    raise SystemExit(
        "The installed PyTorch build cannot access CUDA. Check that its CUDA "
        "runtime is not newer than the maximum CUDA version shown by nvidia-smi."
    )
print("GPU:", torch.cuda.get_device_name(0))
print(
    "GPU memory GiB:",
    round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2),
)

# Exercise the CUDA operations used by Hessian collection and LeanQuant.
device = torch.device("cuda:0")
matrix = torch.randn(256, 256, device=device, dtype=torch.float32)
gram = matrix @ matrix.t()
positive_definite = gram + torch.eye(256, device=device) * 1e-3
factor = torch.linalg.cholesky(positive_definite)
if not torch.isfinite(factor).all():
    raise SystemExit("CUDA Cholesky smoke test produced non-finite values")
torch.cuda.synchronize()
print("CUDA GEMM/Cholesky smoke test: OK")
PY

echo "Testing upstream SqueezeLLM KMeans ..."
"$VENV_DIR/bin/python" - <<'PY'
import numpy as np

from quantizers.upstream_imports import load_squeezellm_kmeans

kmeans_fit = load_squeezellm_kmeans()
values = np.linspace(-1.0, 1.0, 32, dtype=np.float32).reshape(-1, 1)
weights = np.ones(32, dtype=np.float32)
centers, labels = kmeans_fit((values, weights, 8))
if centers.shape != (8,) or labels.shape != (32,):
    raise SystemExit(
        f"Unexpected SqueezeLLM KMeans output: {centers.shape}, {labels.shape}"
    )
print("SqueezeLLM KMeans smoke test: OK")
PY

echo "Testing lm-eval PIQA dataset compatibility ..."
"$VENV_DIR/bin/python" lm_eval_dataset_smoke.py --download-piqa

echo "Server setup complete."
