#!/usr/bin/env bash
set -euo pipefail

# LeanQuant exponent sweep for RTN only on a Linux GPU server.
# Runs exponents {2, 3, 6} for 4-bit and 3-bit, then aggregates PPL/lm-eval.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-server}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
EXPONENT_VALUES="${EXPONENT_VALUES:-2 3 6}"
SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-$ROOT_DIR/outputs/leanquant_exponent_sweep}"
LOG_DIR="${LOG_DIR:-$SWEEP_OUTPUT_ROOT/logs}"

mkdir -p "$SWEEP_OUTPUT_ROOT/runs" "$LOG_DIR"

exponent_tag() {
  printf '%s' "$1" | tr '.' 'p'
}

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/leanquant_exponent_sweep_${TIMESTAMP}.log"
first_run=1

{
  echo "=== LeanQuant RTN exponent sweep ==="
  echo "Exponents: $EXPONENT_VALUES"
  echo "Bits: 4 3"
  echo "Method: RTN only"
  echo "Percdamp: ${LEANQUANT_PERCDAMP:-0.1}"
  echo "Sweep output: $SWEEP_OUTPUT_ROOT"
} | tee -a "$LOG_FILE"

for exponent in $EXPONENT_VALUES; do
  tag="$(exponent_tag "$exponent")"
  run_output="$SWEEP_OUTPUT_ROOT/runs/exponent_${tag}"
  # LeanQuant cache metadata includes exponent, so each exponent keeps an
  # isolated statistics cache. HF/dataset/evaluation caches still follow the
  # shared defaults from run_server_leanquant.sh.
  run_statistics="$run_output/_statistics"
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
    echo "=== Exponent $exponent ==="
    BITS="4 3" \
    METHODS="rtn" \
    LEANQUANT_EXPONENT="$exponent" \
    OUTPUT_ROOT="$run_output" \
    STATISTICS_CACHE_DIR="$run_statistics" \
    LOG_DIR="$run_output/logs" \
    LM_EVAL_OUTPUT_DIR="$run_output/lm_eval" \
    RUN_SETUP="$setup_value" \
    RUN_TESTS="$tests_value" \
    RUN_PREFLIGHT="$preflight_value" \
    bash bash/run_server_leanquant.sh
  } 2>&1 | tee -a "$LOG_FILE"
done

"$PYTHON_BIN" - "$SWEEP_OUTPUT_ROOT" <<'PY'
import csv
import json
import re
import sys
from pathlib import Path

root = Path(sys.argv[1])
rows = []
pattern = re.compile(r"exponent_(.+)$")
for path in sorted((root / "runs").glob("exponent_*/benchmark_results.csv")):
    match = pattern.search(path.parent.name)
    exponent = match.group(1).replace("p", ".") if match else ""
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("codebook") == "LeanQuant" and row.get("method") == "RTN":
                row = {"leanquant-exponent": exponent, **row}
                rows.append(row)

if not rows:
    raise SystemExit(f"No LeanQuant RTN results found under {root / 'runs'}")

rows.sort(key=lambda row: (float(row["leanquant-exponent"]), int(row["bits"])))
fieldnames = ["leanquant-exponent"] + [
    name for name in rows[0] if name != "leanquant-exponent"
]

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

print("\nLeanQuant RTN exponent sweep results")
print("\t".join(fieldnames))
for row in rows:
    print("\t".join(row.get(column, "") for column in fieldnames))
PY

echo "Sweep complete."
echo "Results: $SWEEP_OUTPUT_ROOT/benchmark_results.csv"
echo "Markdown: $SWEEP_OUTPUT_ROOT/benchmark_results.md"
echo "Log: $LOG_FILE"
