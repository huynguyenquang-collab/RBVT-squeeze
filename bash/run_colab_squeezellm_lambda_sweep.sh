#!/usr/bin/env bash
set -euo pipefail

# Colab sweep for SqueezeLLM 3-bit RBVT lambda tuning.
# Runs one RTN baseline and three RBVT jobs with lambda in {0.1, 0.5, 2.0},
# then selects the best RBVT run using the requested preference:
# lower ppl first, higher lm-eval on every task second.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv-colab}"
PYTHON_BIN="${PYTHON_BIN:-$VENV_DIR/bin/python}"

MODEL="${MODEL:-meta-llama/Llama-3.1-8B}"
DEVICE="${DEVICE:-cuda:0}"
BITS="${BITS:-3}"
LAMBDA_VALUES="${LAMBDA_VALUES:-0.1 0.5 2}"

LOCAL_OUTPUT_ROOT="${LOCAL_OUTPUT_ROOT:-/content/rbvtquant_outputs/squeezellm_lambda_sweep}"
DRIVE_OUTPUT_ROOT="${DRIVE_OUTPUT_ROOT:-/content/drive/MyDrive/RBVTQuant/squeezellm_lambda_sweep}"
USE_DRIVE="${USE_DRIVE:-auto}"
LOG_DIR="${LOG_DIR:-}"

CALIB_DATASET="${CALIB_DATASET:-c4}"
N_CALIB="${N_CALIB:-128}"
MAX_LENGTH="${MAX_LENGTH:-2048}"
SEED="${SEED:-42}"

GROUP_SIZE="${GROUP_SIZE:--1}"
ROW_CHUNK="${ROW_CHUNK:-1024}"
SQUEEZELLM_FISHER_SAMPLES="${SQUEEZELLM_FISHER_SAMPLES:-100}"
SQUEEZELLM_FISHER_LENGTH="${SQUEEZELLM_FISHER_LENGTH:-512}"
STATISTICS_CACHE_DIR="${STATISTICS_CACHE_DIR:-$LOCAL_OUTPUT_ROOT/_statistics}"

RBVT_TOPK="${RBVT_TOPK:-0}"
GAP_FLOOR="${GAP_FLOOR:-1e-8}"

EVAL_STRIDE="${EVAL_STRIDE:-512}"
EVAL_MAX_LENGTH="${EVAL_MAX_LENGTH:-2048}"
EVAL_SAMPLES="${EVAL_SAMPLES:-2000}"
LM_EVAL_BATCH_SIZE="${LM_EVAL_BATCH_SIZE:-auto}"
LM_EVAL_NUM_FEWSHOT="${LM_EVAL_NUM_FEWSHOT:-}"
LM_EVAL_LIMIT="${LM_EVAL_LIMIT:-}"
INCLUDE_LM_EVAL="${INCLUDE_LM_EVAL:-1}"
KEEP_MODEL="${KEEP_MODEL:-0}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
RUN_RTN="${RUN_RTN:-1}"
SELECT_BEST="${SELECT_BEST:-1}"

HF_HOME="${HF_HOME:-/content/huggingface}"
HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/content/huggingface/datasets}"
TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-/content/huggingface/transformers}"
EVAL_CACHE_DIR="${EVAL_CACHE_DIR:-$ROOT_DIR/dataset_cache}"
LM_EVAL_OUTPUT_DIR="${LM_EVAL_OUTPUT_DIR:-$LOCAL_OUTPUT_ROOT/lm_eval}"

export HF_HOME
export HF_DATASETS_CACHE
export TRANSFORMERS_CACHE
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "Error: Python environment not found at $PYTHON_BIN." >&2
  echo "Run: bash bash/setup_colab_codebooks.sh" >&2
  exit 1
fi

if [ -z "${HF_TOKEN:-${HUGGINGFACE_HUB_TOKEN:-${HUGGINGFACE_TOKEN:-}}}" ]; then
  echo "Error: HF_TOKEN is required for $MODEL." >&2
  exit 1
fi

if [ "$USE_DRIVE" = "auto" ]; then
  if [ -d /content/drive/MyDrive ]; then
    USE_DRIVE=1
  else
    USE_DRIVE=0
  fi
fi
if [ "$USE_DRIVE" != "0" ] && [ "$USE_DRIVE" != "1" ]; then
  echo "Error: USE_DRIVE must be auto, 0, or 1; got $USE_DRIVE." >&2
  exit 1
fi

if [ "$USE_DRIVE" = "1" ] && [ ! -d /content/drive/MyDrive ]; then
  echo "Error: Google Drive is not mounted at /content/drive/MyDrive." >&2
  echo "Mount Drive in a Colab cell or run with USE_DRIVE=0." >&2
  exit 1
fi

if [ "$RUN_PREFLIGHT" = "1" ]; then
  MODEL="$MODEL" \
  PYTHON_BIN="$PYTHON_BIN" \
  bash bash/check_colab_codebooks.sh
fi

if [ -z "$LOG_DIR" ]; then
  if [ "$USE_DRIVE" = "1" ]; then
    LOG_DIR="$DRIVE_OUTPUT_ROOT/logs"
  else
    LOG_DIR="$LOCAL_OUTPUT_ROOT/logs"
  fi
fi

mkdir -p \
  "$LOCAL_OUTPUT_ROOT" \
  "$HF_HOME" \
  "$HF_DATASETS_CACHE" \
  "$TRANSFORMERS_CACHE" \
  "$EVAL_CACHE_DIR" \
  "$LM_EVAL_OUTPUT_DIR" \
  "$STATISTICS_CACHE_DIR"

if [ "$USE_DRIVE" = "1" ]; then
  mkdir -p "$DRIVE_OUTPUT_ROOT" "$LOG_DIR"
  if [ -d "$DRIVE_OUTPUT_ROOT" ]; then
    cp -a "$DRIVE_OUTPUT_ROOT/." "$LOCAL_OUTPUT_ROOT/" 2>/dev/null || true
  fi
else
  mkdir -p "$LOG_DIR"
fi

TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="$LOG_DIR/squeezellm_lambda_sweep_${TIMESTAMP}.log"
RUNS_DIR="$LOCAL_OUTPUT_ROOT/runs"
mkdir -p "$RUNS_DIR"

lambda_tag() {
  local value="$1"
  echo "$value" | tr '.' 'p'
}

sync_results() {
  if [ "$USE_DRIVE" = "1" ]; then
    mkdir -p "$DRIVE_OUTPUT_ROOT"
    cp -a "$LOCAL_OUTPUT_ROOT/." "$DRIVE_OUTPUT_ROOT/"
  fi
}

run_benchmark() {
  local method="$1"
  local lambda_value="$2"
  local run_label="$3"
  local run_tag
  run_tag="$(lambda_tag "$lambda_value")"
  local run_output_root="$RUNS_DIR/squeezellm_3bit_${method}_l${run_tag}"
  {
    echo
    echo "=== $run_label | method=$method | lambda=$lambda_value ==="
    "$PYTHON_BIN" codebook_benchmark.py \
      --model-path "$MODEL" \
      --device "$DEVICE" \
      --output-root "$run_output_root" \
      --resume \
      --codebooks squeezellm \
      --bits "$BITS" \
      --methods "$method" \
      --calib-dataset "$CALIB_DATASET" \
      --n-calib "$N_CALIB" \
      --max-length "$MAX_LENGTH" \
      --seed "$SEED" \
      --group-size "$GROUP_SIZE" \
      --row-chunk "$ROW_CHUNK" \
      --squeezellm-fisher-samples "$SQUEEZELLM_FISHER_SAMPLES" \
      --squeezellm-fisher-length "$SQUEEZELLM_FISHER_LENGTH" \
      --statistics-cache-dir "$STATISTICS_CACHE_DIR" \
      --rbvt-lambda "$lambda_value" \
      --rbvt-topk "$RBVT_TOPK" \
      --gap-floor "$GAP_FLOOR" \
      --eval-stride "$EVAL_STRIDE" \
      --eval-max-length "$EVAL_MAX_LENGTH" \
      --eval-samples "$EVAL_SAMPLES" \
      --eval-cache-dir "$EVAL_CACHE_DIR" \
      --lm-eval-batch-size "$LM_EVAL_BATCH_SIZE" \
      --lm-eval-output-dir "$LM_EVAL_OUTPUT_DIR"
    if [ -n "$LM_EVAL_NUM_FEWSHOT" ]; then
      echo "Note: lm_eval fewshot is controlled by codebook_benchmark defaults or env."
    fi
  } 2>&1 | tee -a "$LOG_FILE"
}

{
  echo "=== RBVTQuant SqueezeLLM lambda sweep ==="
  echo "Model: $MODEL"
  echo "Device: $DEVICE"
  echo "Bits: $BITS"
  echo "Lambdas: $LAMBDA_VALUES"
  echo "Local output: $LOCAL_OUTPUT_ROOT"
  echo "Drive output: $DRIVE_OUTPUT_ROOT"
  echo "Statistics cache: $STATISTICS_CACHE_DIR"
} | tee -a "$LOG_FILE"

if [ "$RUN_RTN" = "1" ]; then
  BASELINE_CSV="$RUNS_DIR/squeezellm_3bit_rtn/benchmark_results.csv"
  if [ -f "$BASELINE_CSV" ] && "$PYTHON_BIN" - "$BASELINE_CSV" <<'PY'
import csv
import sys

path = sys.argv[1]
with open(path, newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
has_rtn = any(
    r.get("codebook") == "SqueezeLLM"
    and r.get("bits") == "3"
    and r.get("method") == "RTN"
    for r in rows
)
raise SystemExit(0 if has_rtn else 1)
PY
  then
    echo "Reusing existing RTN baseline from $BASELINE_CSV" | tee -a "$LOG_FILE"
  else
    run_benchmark rtn 0 "RTN baseline"
  fi
else
  echo "Skipping RTN baseline." | tee -a "$LOG_FILE"
fi

for lambda_value in $LAMBDA_VALUES; do
  run_benchmark rbvt "$lambda_value" "RBVT sweep"
done

RESULTS_JSON="$LOCAL_OUTPUT_ROOT/benchmark_results.json"
RESULTS_CSV="$LOCAL_OUTPUT_ROOT/benchmark_results.csv"
RESULTS_MD="$LOCAL_OUTPUT_ROOT/benchmark_results.md"

"$PYTHON_BIN" - "$RUNS_DIR" "$RESULTS_JSON" "$RESULTS_CSV" "$RESULTS_MD" <<'PY'
import csv
import json
import sys
from pathlib import Path

runs_dir = Path(sys.argv[1])
json_path = Path(sys.argv[2])
csv_path = Path(sys.argv[3])
md_path = Path(sys.argv[4])

rows = []
for result_csv in sorted(runs_dir.glob("*/benchmark_results.csv")):
    with result_csv.open(newline="", encoding="utf-8") as handle:
        rows.extend(list(csv.DictReader(handle)))

if not rows:
    raise SystemExit(f"No benchmark_results.csv files found under {runs_dir}")

fieldnames = [
    "model",
    "codebook",
    "bits",
    "method",
    "rbvt-lambda",
    "ppl-wiki",
    "ppl-c4",
    "arc-c",
    "arc-e",
    "boolq",
    "hellaswag",
    "lambada",
    "openbookqa",
    "piqa",
    "rte",
    "winogrande",
    "avg",
]

json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
with csv_path.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

md_lines = [
    "| " + " | ".join(fieldnames) + " |",
    "|" + "|".join(["---"] * len(fieldnames)) + "|",
]
for row in rows:
    md_lines.append("| " + " | ".join(row.get(col, "") for col in fieldnames) + " |")
md_path.write_text("\n".join(md_lines) + "\n", encoding="utf-8")

print(f"Wrote aggregate report to {json_path}, {csv_path}, {md_path}")
PY

if [ ! -f "$RESULTS_CSV" ]; then
  echo "Error: benchmark_results.csv not found at $RESULTS_CSV" >&2
  exit 1
fi

if [ "$SELECT_BEST" = "1" ]; then
  "$PYTHON_BIN" - "$RESULTS_CSV" <<'PY'
import csv
import sys
from math import inf

path = sys.argv[1]
rows = []
with open(path, newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        if row.get("codebook") != "SqueezeLLM" or row.get("bits") != "3":
            continue
        if row.get("method") not in {"RTN", "RBVT"}:
            continue
        rows.append(row)

def to_float(value):
    try:
        return float(value)
    except Exception:
        return inf

TASKS = [
    "arc-c",
    "arc-e",
    "boolq",
    "hellaswag",
    "lambada",
    "openbookqa",
    "piqa",
    "rte",
    "winogrande",
]

rtn = [r for r in rows if r["method"] == "RTN"]
rbvt = [r for r in rows if r["method"] == "RBVT"]
if not rtn or not rbvt:
    raise SystemExit("Missing RTN or RBVT rows in benchmark_results.csv")

rtn = rtn[0]
best = None
best_score = None
for row in rbvt:
    task_ok = all(to_float(row[t]) > to_float(rtn[t]) for t in TASKS)
    ppl_ok = (
        to_float(row["ppl-wiki"]) < to_float(rtn["ppl-wiki"])
        and to_float(row["ppl-c4"]) < to_float(rtn["ppl-c4"])
    )
    if not (task_ok and ppl_ok):
        continue
    score = (
        to_float(row["ppl-wiki"]),
        to_float(row["ppl-c4"]),
        -to_float(row["avg"]),
    )
    if best is None or score < best_score:
        best = row
        best_score = score

print("=== Sweep selection ===")
print(
    f"RTN: ppl-wiki={rtn['ppl-wiki']} ppl-c4={rtn['ppl-c4']} avg={rtn['avg']}"
)
for row in rbvt:
    print(
        f"lambda={row['rbvt-lambda']} "
        f"ppl-wiki={row['ppl-wiki']} ppl-c4={row['ppl-c4']} avg={row['avg']}"
    )
if best is None:
    print("No RBVT lambda satisfied the strict selection criteria.")
    sys.exit(0)
print(
    f"Best RBVT row: lambda={best['rbvt-lambda']} bits={best['bits']} "
    f"ppl-wiki={best['ppl-wiki']} ppl-c4={best['ppl-c4']} avg={best['avg']}"
)
PY
else
  echo "Selection disabled; reporting all lambda results only." | tee -a "$LOG_FILE"
fi

sync_results

echo "Sweep complete."
echo "Results: $RESULTS_CSV"
echo "Log: $LOG_FILE"
