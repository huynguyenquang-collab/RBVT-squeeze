#!/usr/bin/env bash
set -euo pipefail

# LeanQuant multi-model benchmark with the best percdamp sweep setting:
# exponent=4.0, percdamp=0.15, bits 4/3, methods RTN/RBVT.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

if [ -f "$ROOT_DIR/.env" ]; then
  set -a
  # shellcheck disable=SC1091
  . "$ROOT_DIR/.env"
  set +a
fi

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-server}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"
SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-$ROOT_DIR/outputs/leanquant_percdamp015_multimodel}"
LOG_DIR="${LOG_DIR:-$SWEEP_OUTPUT_ROOT/logs}"

LEANQUANT_EXPONENT="${LEANQUANT_EXPONENT:-4.0}"
LEANQUANT_PERCDAMP="${LEANQUANT_PERCDAMP:-0.15}"
MODEL_SPECS="${MODEL_SPECS:-Llama31=meta-llama/Llama-3.1-8B;Mistral7Bv03=mistralai/Mistral-7B-v0.3;Qwen25_7B=Qwen/Qwen2.5-7B}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-rbvtquant}"
WANDB_ENTITY="${WANDB_ENTITY:-}"
LM_EVAL_TASKS="${LM_EVAL_TASKS:-arc_challenge arc_easy boolq hellaswag lambada_openai openbookqa piqa rte winogrande mmlu gsm8k}"

mkdir -p "$SWEEP_OUTPUT_ROOT/runs" "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/leanquant_percdamp015_multimodel_${TIMESTAMP}.log"
first_run=1

IFS=';' read -r -a MODEL_ARRAY <<< "$MODEL_SPECS"

{
  echo "=== LeanQuant percdamp=0.15 multi-model benchmark ==="
  echo "Model specs: $MODEL_SPECS"
  echo "Bits: 4 3"
  echo "Methods: RTN/RBVT"
  echo "LM-eval tasks: $LM_EVAL_TASKS"
  echo "LeanQuant exponent: $LEANQUANT_EXPONENT"
  echo "LeanQuant percdamp: $LEANQUANT_PERCDAMP"
  echo "W&B logging: $USE_WANDB | project=$WANDB_PROJECT | entity=${WANDB_ENTITY:-default}"
  echo "Output: $SWEEP_OUTPUT_ROOT"
} | tee -a "$LOG_FILE"

for spec in "${MODEL_ARRAY[@]}"; do
  if [[ "$spec" != *=* ]]; then
    echo "Error: MODEL_SPECS entries must be label=checkpoint; got $spec" >&2
    exit 1
  fi
  label="${spec%%=*}"
  model="${spec#*=}"
  run_output="$SWEEP_OUTPUT_ROOT/runs/$label"
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
    echo "=== Model $label | $model ==="
    MODEL="$model" \
    BITS="4 3" \
    METHODS="rtn rbvt" \
    LEANQUANT_EXPONENT="$LEANQUANT_EXPONENT" \
    LEANQUANT_PERCDAMP="$LEANQUANT_PERCDAMP" \
    OUTPUT_ROOT="$run_output" \
    STATISTICS_CACHE_DIR="$run_statistics" \
    LOG_DIR="$run_output/logs" \
    LM_EVAL_OUTPUT_DIR="$run_output/lm_eval" \
    LM_EVAL_TASKS="$LM_EVAL_TASKS" \
    CLEAN_STATISTICS_CACHE=1 \
    USE_WANDB="$USE_WANDB" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_ENTITY="$WANDB_ENTITY" \
    RUN_SETUP="$setup_value" \
    RUN_TESTS="$tests_value" \
    RUN_PREFLIGHT="$preflight_value" \
    bash bash/run_server_leanquant.sh
  } 2>&1 | tee -a "$LOG_FILE"
done

"$PYTHON_BIN" - "$SWEEP_OUTPUT_ROOT" "$LEANQUANT_EXPONENT" "$LEANQUANT_PERCDAMP" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
exponent = sys.argv[2]
percdamp = sys.argv[3]
rows = []
for path in sorted((root / "runs").glob("*/benchmark_results.csv")):
    label = path.parent.name
    checkpoint = ""
    summary_paths = sorted(path.parent.glob("*/run_summary.json"))
    if summary_paths:
        try:
            checkpoint = json.loads(summary_paths[0].read_text(encoding="utf-8"))[
                "model_path"
            ]
        except Exception:
            checkpoint = ""
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("codebook") == "LeanQuant":
                rows.append(
                    {
                        "model-key": label,
                        "checkpoint": checkpoint,
                        "leanquant-exponent": exponent,
                        "leanquant-percdamp": percdamp,
                        **row,
                    }
                )

if not rows:
    raise SystemExit(f"No LeanQuant results found under {root / 'runs'}")

method_order = {"RTN": 0, "RBVT": 1}
bit_order = {"4": 0, "3": 1}
rows.sort(
    key=lambda row: (
        row["model-key"],
        bit_order.get(row["bits"], 99),
        method_order.get(row["method"], 99),
    )
)
fieldnames = [
    "model-key",
    "checkpoint",
    "leanquant-exponent",
    "leanquant-percdamp",
] + [
    name
    for name in rows[0]
    if name
    not in {
        "model-key",
        "checkpoint",
        "leanquant-exponent",
        "leanquant-percdamp",
    }
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

print("\nLeanQuant percdamp=0.15 multi-model results")
print("\t".join(fieldnames))
for row in rows:
    print("\t".join(row.get(column, "") for column in fieldnames))
PY

echo "Benchmark complete."
echo "Results: $SWEEP_OUTPUT_ROOT/benchmark_results.csv"
echo "Markdown: $SWEEP_OUTPUT_ROOT/benchmark_results.md"
echo "Log: $LOG_FILE"
