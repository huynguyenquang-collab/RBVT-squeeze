#!/usr/bin/env bash
set -euo pipefail

# Remove LeanQuant exponent sweep outputs/statistics on a server.
# Dry-run by default. Set CONFIRM=1 to delete.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-$ROOT_DIR/outputs/leanquant_exponent_sweep}"
CONFIRM="${CONFIRM:-0}"

echo "LeanQuant exponent sweep path:"
echo "  $SWEEP_OUTPUT_ROOT"

if [ -e "$SWEEP_OUTPUT_ROOT" ]; then
  du -sh "$SWEEP_OUTPUT_ROOT" || true
else
  echo "Path does not exist; nothing to clean."
  exit 0
fi

if [ "$CONFIRM" != "1" ]; then
  echo
  echo "Dry run only. To delete, run:"
  echo "  CONFIRM=1 bash bash/clean_server_leanquant_exponent_sweep.sh"
  exit 0
fi

rm -rf "$SWEEP_OUTPUT_ROOT"
echo "Deleted: $SWEEP_OUTPUT_ROOT"
df -h .
