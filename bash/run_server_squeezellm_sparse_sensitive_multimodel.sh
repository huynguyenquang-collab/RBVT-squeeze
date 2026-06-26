#!/usr/bin/env bash
set -euo pipefail

# SqueezeLLM dense+sparse+sensitive multi-model benchmark:
# bits 4/3, methods RTN/RBVT, full PPL + lm-eval task set.

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
SWEEP_OUTPUT_ROOT="${SWEEP_OUTPUT_ROOT:-$ROOT_DIR/outputs/squeezellm_sparse_sensitive_multimodel}"
LOG_DIR="${LOG_DIR:-$SWEEP_OUTPUT_ROOT/logs}"
MODEL_SPECS="${MODEL_SPECS:-Llama31=meta-llama/Llama-3.1-8B;Mistral7Bv03=mistralai/Mistral-7B-v0.3;Qwen25_7B=Qwen/Qwen2.5-7B}"

SQUEEZELLM_OUTLIER_RANGE="${SQUEEZELLM_OUTLIER_RANGE:-1.8}"
SQUEEZELLM_SENSITIVE_PERCENT="${SQUEEZELLM_SENSITIVE_PERCENT:-0.05}"
SPARSE_DEVICE="${SPARSE_DEVICE:-cuda:1}"
LM_EVAL_TASKS="${LM_EVAL_TASKS:-arc_challenge arc_easy boolq hellaswag lambada_openai openbookqa piqa rte winogrande mmlu gsm8k}"
USE_WANDB="${USE_WANDB:-1}"
WANDB_PROJECT="${WANDB_PROJECT:-RBVTsqueeze}"
WANDB_ENTITY="${WANDB_ENTITY:-}"

mkdir -p "$SWEEP_OUTPUT_ROOT/runs" "$LOG_DIR"

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/squeezellm_sparse_sensitive_multimodel_${TIMESTAMP}.log"
first_run=1

IFS=';' read -r -a MODEL_ARRAY <<< "$MODEL_SPECS"

{
  echo "=== SqueezeLLM sparse+sensitive multi-model benchmark ==="
  echo "Model specs: $MODEL_SPECS"
  echo "Device: $SPARSE_DEVICE"
  echo "Bits: 4 3"
  echo "Methods: RTN/RBVT"
  echo "LM-eval tasks: $LM_EVAL_TASKS"
  echo "Outlier range: $SQUEEZELLM_OUTLIER_RANGE"
  echo "Sensitive percent: $SQUEEZELLM_SENSITIVE_PERCENT"
  echo "W&B logging: $USE_WANDB | project=$WANDB_PROJECT | entity=${WANDB_ENTITY:-default}"
  echo "Output: $SWEEP_OUTPUT_ROOT"
  echo "Cache cleanup: disabled; all SqueezeLLM caches are kept"
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
    preflight_value="${RUN_PREFLIGHT:-0}"
    first_run=0
  fi

  {
    echo
    echo "=== Model $label | $model ==="
    MODEL="$model" \
    DEVICE="$SPARSE_DEVICE" \
    BITS="4 3" \
    METHODS="rtn rbvt" \
    LM_EVAL_TASKS="$LM_EVAL_TASKS" \
    SQUEEZELLM_MODE="hybrid" \
    SQUEEZELLM_OUTLIER_RANGE="$SQUEEZELLM_OUTLIER_RANGE" \
    SQUEEZELLM_SENSITIVE_PERCENT="$SQUEEZELLM_SENSITIVE_PERCENT" \
    OUTPUT_ROOT="$run_output" \
    STATISTICS_CACHE_DIR="$run_statistics" \
    LOG_DIR="$run_output/logs" \
    LM_EVAL_OUTPUT_DIR="$run_output/lm_eval" \
    CLEAN_STATISTICS_CACHE=1 \
    USE_WANDB="$USE_WANDB" \
    WANDB_PROJECT="$WANDB_PROJECT" \
    WANDB_ENTITY="$WANDB_ENTITY" \
    PYTHON_BIN="$PYTHON_BIN" \
    RUN_SETUP="$setup_value" \
    RUN_TESTS="$tests_value" \
    RUN_PREFLIGHT="$preflight_value" \
    bash bash/run_server_squeezellm.sh
  } 2>&1 | tee -a "$LOG_FILE"
done

"$PYTHON_BIN" - "$SWEEP_OUTPUT_ROOT" "$SQUEEZELLM_OUTLIER_RANGE" "$SQUEEZELLM_SENSITIVE_PERCENT" <<'PY'
import csv
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
outlier_range = sys.argv[2]
sensitive_percent = sys.argv[3]
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
            if row.get("codebook") == "SqueezeLLM":
                rows.append(
                    {
                        "model-key": label,
                        "checkpoint": checkpoint,
                        "squeezellm-mode": "sparse-sensitive",
                        "squeezellm-outlier-range": outlier_range,
                        "squeezellm-sensitive-percent": sensitive_percent,
                        **row,
                    }
                )

if not rows:
    raise SystemExit(f"No SqueezeLLM results found under {root / 'runs'}")

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
    "squeezellm-mode",
    "squeezellm-outlier-range",
    "squeezellm-sensitive-percent",
] + [
    name
    for name in rows[0]
    if name
    not in {
        "model-key",
        "checkpoint",
        "squeezellm-mode",
        "squeezellm-outlier-range",
        "squeezellm-sensitive-percent",
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

print("\nSqueezeLLM sparse+sensitive multi-model results")
print("\t".join(fieldnames))
for row in rows:
    print("\t".join(row.get(column, "") for column in fieldnames))
PY

echo "Benchmark complete."
echo "Results: $SWEEP_OUTPUT_ROOT/benchmark_results.csv"
echo "Markdown: $SWEEP_OUTPUT_ROOT/benchmark_results.md"
echo "Log: $LOG_FILE"
