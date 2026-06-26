"""
lm-evaluation-harness integration for RBVTQuant.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from runtime_utils import is_numeric_metric_value


class LMEvalHarnessRunner:
    def __init__(
        self,
        tasks: list[str],
        device: str = "cuda",
        batch_size: str = "auto",
        num_fewshot: int | None = None,
        limit: int | float | None = None,
        output_dir: str = "./outputs/lm_eval",
        run_name: str | None = None,
        hf_token: str | None = None,
    ):
        self.tasks = tasks
        self.device = device
        self.batch_size = batch_size
        self.num_fewshot = num_fewshot
        self.limit = limit
        self.output_dir = Path(output_dir)
        self.run_name = run_name or datetime.now().strftime("%Y%m%d-%H%M%S")
        self.hf_token = hf_token
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _model_args(self, model_path: str) -> str:
        dtype = "float16" if self.device.startswith("cuda") else "float32"
        model_args = f"pretrained={model_path},dtype={dtype},trust_remote_code=True"
        if self.hf_token:
            model_args += f",token={self.hf_token}"
        return model_args

    def _make_json_safe(self, value):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, dict):
            return {str(key): self._make_json_safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [self._make_json_safe(item) for item in value]

        item_method = getattr(value, "item", None)
        if callable(item_method):
            try:
                return self._make_json_safe(item_method())
            except (TypeError, ValueError):
                pass

        return repr(value)

    def _summarize_results(self, payload: dict) -> dict:
        summary = {}
        for section_name in ("results", "groups"):
            section = payload.get(section_name, {})
            if not isinstance(section, dict):
                continue
            for task_name, metrics in section.items():
                if not isinstance(metrics, dict):
                    continue
                task_summary = summary.setdefault(task_name, {})
                for metric_name, value in metrics.items():
                    if is_numeric_metric_value(value):
                        task_summary[metric_name] = value
        return summary

    def _write_raw_results(self, model_name: str, payload: dict):
        output_path = self.output_dir / f"{self.run_name}_{model_name}.json"
        with open(output_path, "w", encoding="utf-8") as handle:
            json.dump(self._make_json_safe(payload), handle, indent=2, sort_keys=True)

    def _patch_datasets_repo_aliases(self):
        """Map legacy short dataset ids to canonical HF repo ids.

        Recent lm-eval + transformers versions no longer need the older
        AutoModelForVision2Seq workaround, but some environments still resolve
        task datasets through short ids such as ``winogrande`` or ``boolq``.
        Normalizing them here keeps the harness path stable across machines.
        """
        try:
            import datasets
            import datasets.load as datasets_load
        except ImportError:
            return

        aliases = {
            "wikitext": "Salesforce/wikitext",
            "ai2_arc": "allenai/ai2_arc",
            "hellaswag": "Rowan/hellaswag",
            "piqa": "ybisk/piqa",
            "winogrande": "allenai/winogrande",
            "openbookqa": "allenai/openbookqa",
            "boolq": "google/boolq",
            "glue": "nyu-mll/glue",
            "super_glue": "aps/super_glue",
            "lambada_openai": "EleutherAI/lambada_openai",
        }

        def normalize(value):
            if isinstance(value, str):
                return aliases.get(value, value)
            return value

        def patch_callable(module, name: str, key_names: tuple[str, ...]):
            original = getattr(module, name, None)
            if original is None or getattr(original, "_rbvt_patched", False):
                return

            def wrapper(*args, **kwargs):
                new_args = list(args)
                if new_args:
                    new_args[0] = normalize(new_args[0])
                for key in key_names:
                    if key in kwargs:
                        kwargs[key] = normalize(kwargs[key])
                return original(*new_args, **kwargs)

            wrapper._rbvt_patched = True  # type: ignore[attr-defined]
            setattr(module, name, wrapper)

        patch_callable(datasets, "load_dataset", ("path", "path_or_name"))
        patch_callable(datasets, "load_dataset_builder", ("path", "path_or_name"))
        patch_callable(datasets, "get_dataset_config_names", ("path", "path_or_name"))
        patch_callable(datasets_load, "load_dataset", ("path", "path_or_name"))
        patch_callable(datasets_load, "load_dataset_builder", ("path", "path_or_name"))
        patch_callable(datasets_load, "get_dataset_config_names", ("path", "path_or_name"))
        patch_callable(datasets_load, "dataset_module_factory", ("path", "path_or_name"))

    def evaluate_model(self, model_name: str, model_path: str) -> dict:
        try:
            self._patch_datasets_repo_aliases()
            from lm_eval import evaluator
        except ImportError as exc:
            raise RuntimeError(
                "lm-eval is not installed. Install the 'lm-eval' package or disable lm-eval with --no-lm-eval."
            ) from exc

        payload = evaluator.simple_evaluate(
            model="hf",
            model_args=self._model_args(model_path),
            tasks=self.tasks,
            device=self.device,
            batch_size=self.batch_size,
            num_fewshot=self.num_fewshot,
            limit=self.limit,
            log_samples=False,
        )
        self._write_raw_results(model_name, payload)
        return {
            "tasks": list(self.tasks),
            "summary": self._summarize_results(payload),
            "raw": self._make_json_safe(payload),
        }

    def run(self, model_paths: dict[str, str]) -> dict:
        results = {}
        for model_name, model_path in model_paths.items():
            print(f"\nRunning lm-eval for {model_name} on {', '.join(self.tasks)}...")
            results[model_name] = self.evaluate_model(model_name, model_path)
        return results
