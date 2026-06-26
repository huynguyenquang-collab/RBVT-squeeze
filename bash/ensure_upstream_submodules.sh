#!/usr/bin/env bash
set -euo pipefail

# Ensure git submodules that are required at runtime are present.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

required_paths=(
  "SqueezeLLM-gradients/datautils.py"
  "SqueezeLLM-gradients/run.py"
)

missing=0
for path in "${required_paths[@]}"; do
  if [ ! -f "$path" ]; then
    missing=1
    break
  fi
done

if [ "$missing" = "0" ]; then
  exit 0
fi

if [ ! -f .gitmodules ]; then
  echo "Error: required upstream files are missing and .gitmodules is absent." >&2
  exit 1
fi
if ! command -v git >/dev/null 2>&1; then
  echo "Error: git is required to initialize upstream submodules." >&2
  exit 1
fi

echo "Initializing required upstream submodules ..."
git submodule update --init --recursive SqueezeLLM-gradients

for path in "${required_paths[@]}"; do
  if [ ! -f "$path" ]; then
    echo "Error: missing required upstream file after submodule init: $path" >&2
    exit 1
  fi
done
