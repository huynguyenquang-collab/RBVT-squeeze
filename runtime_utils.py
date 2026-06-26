from __future__ import annotations

import os
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_ENV_PATH = ROOT / ".env"
LM_EVAL_PREFERRED_METRICS = (
    "acc,none",
    "acc_norm,none",
    "exact_match,strict-match",
    "exact_match,flexible-extract",
    "exact_match,none",
    "exact_match",
)
LM_EVAL_WANDB_CANONICAL_METRICS = {
    "mmlu": (
        ("lm_eval/mmlu", ("acc,none", "acc")),
    ),
    "gsm8k": (
        ("lm_eval/gsm8k_strict", ("exact_match,strict-match",)),
        ("lm_eval/gsm8k_flexible", ("exact_match,flexible-extract",)),
    ),
}

DEFAULT_LM_EVAL_TASKS = {
    "smoke": [
        "piqa",
    ],
    "core": [
        "arc_easy",
        "arc_challenge",
        "hellaswag",
        "piqa",
        "winogrande",
    ],
    "extended": [
        "arc_easy",
        "arc_challenge",
        "hellaswag",
        "piqa",
        "winogrande",
        "boolq",
        "rte",
        "openbookqa",
        "lambada_openai",
    ],
}


def load_runtime_env(env_path: str | Path | None = None):
    env_file = Path(env_path) if env_path is not None else DEFAULT_ENV_PATH
    if not env_file.exists():
        return

    try:
        from dotenv import load_dotenv

        load_dotenv(env_file, override=False)
        return
    except ImportError:
        pass

    for line in env_file.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip("\"'"))


def resolve_hf_token() -> str | None:
    return (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
    )


def resolve_wandb_api_key() -> str | None:
    return os.getenv("WANDB_API_KEY")


def build_model_slug(model_ref: str) -> str:
    candidate = str(model_ref).rstrip("/\\").split("/")[-1].split("\\")[-1]

    def replace_numeric_dot(match: re.Match[str]) -> str:
        start = match.start()
        if start > 0 and candidate[start - 1].lower() == "v":
            return match.group(0)
        return match.group(0).replace(".", "p")

    candidate = re.sub(r"\d+\.\d+", replace_numeric_dot, candidate)
    candidate = candidate.replace(" ", "_").replace("/", "_").replace("\\", "_")
    return candidate


def is_numeric_metric_value(value) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def pick_lm_eval_metric(task_metrics: dict) -> tuple[str | None, float | None]:
    if not isinstance(task_metrics, dict):
        return None, None

    for metric_name in LM_EVAL_PREFERRED_METRICS:
        value = task_metrics.get(metric_name)
        if is_numeric_metric_value(value):
            return metric_name, float(value)

    for metric_name, value in task_metrics.items():
        if metric_name.endswith("_stderr") or metric_name == "alias":
            continue
        if is_numeric_metric_value(value):
            return metric_name, float(value)

    return None, None


def collect_lm_eval_wandb_metrics(task_results: dict) -> dict[str, float]:
    if not isinstance(task_results, dict):
        return {}

    metrics = {}
    for task_name, task_metrics in task_results.items():
        if not isinstance(task_metrics, dict):
            continue
        if task_name != "mmlu" and task_name.startswith("mmlu"):
            continue
        if task_name == "gsm8k":
            continue

        accuracy = task_metrics.get("acc,none")
        if is_numeric_metric_value(accuracy):
            metrics[f"lm_eval/{task_name}"] = float(accuracy)

    for task_name, metric_specs in LM_EVAL_WANDB_CANONICAL_METRICS.items():
        task_metrics = task_results.get(task_name, {})
        if not isinstance(task_metrics, dict):
            continue
        for wandb_key, metric_names in metric_specs:
            for metric_name in metric_names:
                value = task_metrics.get(metric_name)
                if is_numeric_metric_value(value):
                    metrics[wandb_key] = float(value)
                    break
            if wandb_key in metrics:
                continue

    return metrics
