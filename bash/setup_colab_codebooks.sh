#!/usr/bin/env bash
set -euo pipefail

# Install a Python 3.12 environment for the codebook benchmark on Google Colab.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-colab}"
UV_CACHE_DIR="${UV_CACHE_DIR:-/content/.cache/uv}"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/content/.cache/pip}"
PYTHON_VERSION="${PYTHON_VERSION:-3.12}"

export UV_CACHE_DIR
export PIP_CACHE_DIR

echo "=== RBVTQuant Colab setup ==="
echo "Repository: $ROOT_DIR"
echo "Python: $PYTHON_VERSION"
echo "Virtual environment: $VENV_DIR"

echo "Initializing upstream repositories ..."
bash bash/ensure_upstream_submodules.sh

if [ ! -d /content ]; then
  echo "Warning: /content is missing; this does not look like a Google Colab runtime."
fi

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "Error: nvidia-smi is unavailable. In Colab, select a GPU runtime first." >&2
  exit 1
fi

nvidia-smi

if command -v uv >/dev/null 2>&1; then
  UV_BIN="$(command -v uv)"
else
  echo "Installing uv ..."
  python3 -m pip install --quiet --upgrade uv
  UV_BIN="$(python3 -m site --user-base)/bin/uv"
  if [ ! -x "$UV_BIN" ]; then
    UV_BIN="$(command -v uv)"
  fi
fi

echo "Using uv: $UV_BIN"
"$UV_BIN" python install "$PYTHON_VERSION"

if [ ! -x "$VENV_DIR/bin/python" ]; then
  "$UV_BIN" venv --python "$PYTHON_VERSION" "$VENV_DIR"
fi

echo "Installing requirements.txt ..."
"$UV_BIN" pip install \
  --python "$VENV_DIR/bin/python" \
  --upgrade \
  -r requirements.txt

echo "Verifying Python and CUDA ..."
"$VENV_DIR/bin/python" - <<'PY'
import sys

import torch

print("Python:", sys.version.replace("\n", " "))
print("PyTorch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA runtime:", torch.version.cuda)
if not torch.cuda.is_available():
    raise SystemExit(
        "PyTorch cannot access CUDA. Restart with a Colab GPU runtime or install "
        "a CUDA-enabled PyTorch build."
    )
print("GPU:", torch.cuda.get_device_name(0))
properties = torch.cuda.get_device_properties(0)
print("GPU memory GiB:", round(properties.total_memory / 1024**3, 2))
PY

echo
echo "Setup complete."
echo "Python executable: $VENV_DIR/bin/python"
echo "Next: bash bash/check_colab_codebooks.sh"
