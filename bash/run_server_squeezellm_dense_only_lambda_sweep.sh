#!/usr/bin/env bash
set -euo pipefail

# SqueezeLLM dense-only 4-bit RBVT sweep for lambda in {0.1, 0.5, 3}.
# Reuses the Fisher and dense-only LUT cache from run_server_squeezellm_dense_only.sh.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-server}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
LAMBDA_VALUES="${LAMBDA_VALUES:-0.1 0.5 3}"
SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-$ROOT_DIR/outputs/squeezellm_dense_only_lambda_sweep}"
STATISTICS_CACHE_DIR="${STATISTICS_CACHE_DIR:-$ROOT_DIR/outputs/squeezellm_dense_only_server/_statistics}"
LOG_DIR="${LOG_DIR:-$SWEEP_OUTPUT_ROOT/logs}"

mkdir -p "$SWEEP_OUTPUT_ROOT/runs" "$LOG_DIR"

lambda_tag() {
  printf '%s' "$1" | tr '.' 'p'
}

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/squeezellm_dense_only_lambda_sweep_${TIMESTAMP}.log"
first_run=1

{
  echo "=== SqueezeLLM dense-only 4-bit RBVT lambda sweep ==="
  echo "Lambdas: $LAMBDA_VALUES"
  echo "Statistics cache: $STATISTICS_CACHE_DIR"
  echo "Sweep output: $SWEEP_OUTPUT_ROOT"
} | tee -a "$LOG_FILE"

for lambda_value in $LAMBDA_VALUES; do
  tag="$(lambda_tag "$lambda_value")"
  run_output="$SWEEP_OUTPUT_ROOT/runs/lambda_${tag}"
  setup_value=0
  tests_value=0
  preflight_value=0
  if [ "$first_run" = "1" ]; then
    setup_value="${RUN_SETUP:-0}"
    tests_value="${RUN_TESTS:-0}"
    preflight_value="${RUN_PREFLIGHT:-1}"
    first_run=0
  fi

  {
    echo
    echo "=== Lambda $lambda_value ==="
    SQUEEZELLM_MODE="dense-only" \
    BITS="4" \
    METHODS="rbvt" \
    RBVT_LAMBDA="$lambda_value" \
    OUTPUT_ROOT="$run_output" \
    STATISTICS_CACHE_DIR="$STATISTICS_CACHE_DIR" \
    LOG_DIR="$run_output/logs" \
    LM_EVAL_OUTPUT_DIR="$run_output/lm_eval" \
    RUN_SETUP="$setup_value" \
    RUN_TESTS="$tests_value" \
    RUN_PREFLIGHT="$preflight_value" \
    bash bash/run_server_squeezellm.sh
  } 2>&1 | tee -a "$LOG_FILE"
done

"$PYTHON_BIN" - "$SWEEP_OUTPUT_ROOT" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
for path in sorted((root / "runs").glob("lambda_*/benchmark_results.csv")):
    with path.open(newline="", encoding="utf-8") as handle:
        rows.extend(
            row
            for row in csv.DictReader(handle)
            if row.get("bits") == "4" and row.get("method") == "RBVT"
        )

if not rows:
    raise SystemExit(f"No 4-bit RBVT results found under {root / 'runs'}")

rows.sort(key=lambda row: float(row["rbvt-lambda"]))
fieldnames = list(rows[0])

(root / "benchmark_results.json").write_text(
    json.dumps(rows, indent=2),
    encoding="utf-8",
)
with (root / "benchmark_results.csv").open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

lines = [
    "| " + " | ".join(fieldnames) + " |",
    "|" + "|".join(["---"] * len(fieldnames)) + "|",
]
lines.extend(
    "| " + " | ".join(row.get(column, "") for column in fieldnames) + " |"
    for row in rows
)
(root / "benchmark_results.md").write_text(
    "\n".join(lines) + "\n",
    encoding="utf-8",
)

print("\nDense-only 4-bit RBVT sweep results")
print("\t".join(fieldnames))
for row in rows:
    print("\t".join(row.get(column, "") for column in fieldnames))
PY

echo "Sweep complete."
echo "Results: $SWEEP_OUTPUT_ROOT/benchmark_results.csv"
echo "Markdown: $SWEEP_OUTPUT_ROOT/benchmark_results.md"
echo "Log: $LOG_FILE"
