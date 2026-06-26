"""Benchmark LeanQuant and SqueezeLLM codebooks with RTN, RBVT, and GPTQ."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from calibration_utils import load_calibration_data
from lm_eval_runner import LMEvalHarnessRunner
from main import (
    collect_layer_stats,
    cleanup_output_dir,
    evaluate_quantized_model,
    is_lmhead,
)
from nonuniform_gptq import quantize_codebook_model_gptq
from quantizers import apply_rbvt
from quantizers.base_codebook import CodebookContext
from quantizers.codebook_store import CodebookStore
from quantizers.hessian_store import HessianStore
from quantizers.codebook_factory import get_codebook
from quantizers.leanquant_collector import collect_leanquant_codebooks
from quantizers.sensitivity_store import SensitivityStore
from quantizers.sparse_residual_store import SparseResidualStore
from quantizers.squeezellm_collector import (
    collect_squeezellm_codebooks,
    collect_squeezellm_fisher,
    load_squeezellm_fisher_data,
)
from quantizers.upstream_calibration import load_upstream_c4_tokens
from quantizers.upstream_imports import (
    SQUEEZELLM_GRADIENTS_SOURCE,
    load_leanquant_upstream,
)
from runtime_utils import collect_lm_eval_wandb_metrics
from runtime_utils import build_model_slug, load_runtime_env, resolve_hf_token
from runtime_utils import resolve_wandb_api_key


DEFAULT_MODEL = "meta-llama/Llama-3.1-8B"
SQUEEZELLM_HYBRID_SOURCE = (
    f"{SQUEEZELLM_GRADIENTS_SOURCE}+SqueezeLLM-dense-sparse-sensitive-v1"
)
SQUEEZELLM_DENSE_ONLY_SOURCE = (
    f"{SQUEEZELLM_GRADIENTS_SOURCE}+SqueezeLLM-dense-only-v1"
)
DEFAULT_LM_EVAL_TASKS = [
    "arc_challenge",
    "arc_easy",
    "boolq",
    "hellaswag",
    "lambada_openai",
    "openbookqa",
    "piqa",
    "rte",
    "winogrande",
]
RESULT_COLUMNS = [
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
TASK_COLUMNS = {
    "arc-c": ("arc_challenge", ("acc_norm,none", "acc,none")),
    "arc-e": ("arc_easy", ("acc_norm,none", "acc,none")),
    "boolq": ("boolq", ("acc,none",)),
    "hellaswag": ("hellaswag", ("acc_norm,none", "acc,none")),
    "lambada": ("lambada_openai", ("acc,none",)),
    "openbookqa": ("openbookqa", ("acc_norm,none", "acc,none")),
    "piqa": ("piqa", ("acc_norm,none", "acc,none")),
    "rte": ("rte", ("acc,none",)),
    "winogrande": ("winogrande", ("acc,none",)),
}


def _device_map(device: str):
    return {"": device}


def _model_label(model_path: str) -> str:
    name = model_path.rstrip("/").split("/")[-1]
    if name.lower() == "llama-3.1-8b":
        return "Llama31"
    return name


def _set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _print_section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


@torch.no_grad()
def quantize_with_codebook(
    model,
    codebook,
    method: str,
    means: Dict[str, torch.Tensor],
    variances: Dict[str, torch.Tensor],
    skip_lmhead: bool,
    row_chunk: int,
    rbvt_lambda: float,
    rbvt_topk: int,
    gap_floor: float,
    strict_descent: bool,
    codebook_store: CodebookStore,
    sparse_store: SparseResidualStore | None = None,
) -> dict:
    linears: List[Tuple[str, nn.Module]] = [
        (name, module)
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
    ]
    if skip_lmhead:
        linears = [
            (name, module)
            for name, module in linears
            if not is_lmhead(name)
        ]
    print(
        f"Quantizing {len(linears)} Linear layers "
        f"({'skipping' if skip_lmhead else 'including'} lm_head) | method={method}"
    )

    totals = {
        "flips": 0,
        "candidates": 0,
        "boundary_kept": 0,
        "bias_before": 0.0,
        "bias_after": 0.0,
        "objective_before": 0.0,
        "objective_after": 0.0,
        "variance_increase": 0.0,
        "sparse_values": 0,
    }

    for name, module in tqdm(linears, desc="Quantizing layers"):
        weight = module.weight.data
        sparse_residual = None
        sparse_mask = None
        dense_weight = weight
        if sparse_store is not None:
            sparse_residual = sparse_store.get(name, device=weight.device).to(weight.dtype)
            sparse_mask = sparse_residual != 0
            dense_weight = weight - sparse_residual
            totals["sparse_values"] += int(sparse_mask.sum().item())
        cached_centers = codebook_store.get(name)
        codebook.set_context(
            CodebookContext(
                precomputed_centers=cached_centers,
            )
        )
        quantized = codebook.quantize(dense_weight, row_chunk=row_chunk)
        if sparse_mask is not None:
            quantized.W_dequant[sparse_mask] = 0
        output = quantized.W_dequant

        if method == "rbvt":
            if name not in means:
                raise RuntimeError(f"Missing activation mean for RBVT layer {name!r}")
            sigma = variances.get(name)
            output, stats = apply_rbvt(
                W_fp=dense_weight,
                qres=quantized,
                mu=means[name].to(weight.device),
                sigma_ii=sigma.to(weight.device) if sigma is not None else None,
                rbvt_lambda=rbvt_lambda,
                rbvt_topk=rbvt_topk if rbvt_topk > 0 else None,
                row_chunk=row_chunk,
                gap_floor=gap_floor,
                strict_descent=strict_descent,
                candidate_mask=~sparse_mask if sparse_mask is not None else None,
            )
            for key in totals:
                if key == "sparse_values":
                    continue
                totals[key] += getattr(stats, key)

        if sparse_residual is not None:
            output[sparse_mask] = 0
            output = output + sparse_residual
        module.weight.data = output.to(weight.dtype)
        codebook.set_context(None)
        del quantized, output, weight
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    codebook_store.mark_complete()
    if method == "rbvt":
        print(
            "RBVT summary | "
            f"flips={totals['flips']} | candidates={totals['candidates']} | "
            f"boundary_kept={totals['boundary_kept']}"
        )
        print(
            "RBVT objective | "
            f"bias_before={totals['bias_before']:.6e} -> "
            f"bias_after={totals['bias_after']:.6e} | "
            f"objective_before={totals['objective_before']:.6e} -> "
            f"objective_after={totals['objective_after']:.6e} | "
            f"variance_increase={totals['variance_increase']:.6e}"
        )
    else:
        print("RTN summary | plain nearest-codeword quantization completed.")
    if sparse_store is not None:
        print(
            "SqueezeLLM sparse summary | "
            f"restored_values={totals['sparse_values']}"
        )

    result = {
        "method": method,
        "num_linear_layers": len(linears),
        "skip_lmhead": skip_lmhead,
        "codebook": codebook.name,
        "bits": codebook.bits,
        "sparse_values": totals["sparse_values"],
    }
    if method == "rbvt":
        result.update(totals)
        result["rbvt_topk"] = rbvt_topk
    return result


def _extract_task_metric(
    task_results: dict,
    task_name: str,
    metric_names: tuple[str, ...],
) -> float | None:
    metrics = task_results.get(task_name, {})
    if not isinstance(metrics, dict):
        return None
    for metric_name in metric_names:
        value = metrics.get(metric_name)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
    return None


def build_result_row(summary: dict) -> dict:
    evaluation = summary.get("evaluation", {})
    perplexity = evaluation.get("perplexity", {})
    lm_eval = evaluation.get("lm_eval", {})
    run_label = summary["run_label"]
    task_results = lm_eval.get(run_label, {}).get("summary", {})

    row = {
        "model": _model_label(summary["model_path"]),
        "codebook": {
            "leanquant": "LeanQuant",
            "squeezellm": "SqueezeLLM",
        }.get(summary["codebook"], summary["codebook"]),
        "bits": summary["bits"],
        "method": summary["method"].upper(),
        "rbvt-lambda": summary.get("args", {}).get("rbvt_lambda")
        if summary.get("method") == "rbvt"
        else "",
        "ppl-wiki": perplexity.get("WikiText-2", {}).get("perplexity"),
        "ppl-c4": perplexity.get("C4", {}).get("perplexity"),
    }
    accuracies = []
    for column, (task_name, metric_names) in TASK_COLUMNS.items():
        value = _extract_task_metric(task_results, task_name, metric_names)
        row[column] = value
        if value is not None:
            accuracies.append(value)
    row["avg"] = (
        sum(accuracies) / len(accuracies)
        if len(accuracies) == len(TASK_COLUMNS)
        else None
    )
    return row


def _wandb_metric_name(column: str) -> str:
    if column == "ppl-wiki":
        return "perplexity/WikiText-2"
    if column == "ppl-c4":
        return "perplexity/C4"
    if column == "avg":
        return "lm_eval/avg"
    return f"lm_eval/{column}"


def _flatten_lm_eval_metrics(summary: dict) -> dict[str, float]:
    run_label = summary.get("run_label")
    task_summary = (
        summary.get("evaluation", {})
        .get("lm_eval", {})
        .get(run_label, {})
        .get("summary", {})
    )
    if not isinstance(task_summary, dict):
        return {}

    return collect_lm_eval_wandb_metrics(task_summary)


def log_summary_to_wandb(args, summary: dict, run_id: str, elapsed: float):
    try:
        import wandb
    except ImportError:
        print("Warning: wandb is not installed; skipping W&B logging.")
        return

    api_key = resolve_wandb_api_key()
    if api_key:
        try:
            wandb.login(key=api_key, relogin=True)
        except Exception as exc:
            print(f"Warning: wandb login failed; skipping W&B logging: {exc}")
            return

    row = build_result_row(summary)
    metrics = {}
    for column in ("ppl-wiki", "ppl-c4"):
        value = row.get(column)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            metrics[_wandb_metric_name(column)] = value
    metrics.update(_flatten_lm_eval_metrics(summary))
    metrics["runtime/elapsed_seconds"] = elapsed

    codebook = summary.get("codebook")
    bits = summary.get("bits")
    method = summary.get("method")
    model_slug = build_model_slug(summary.get("model_path", args.model_path))
    run_name = f"{codebook}_{model_slug}_{bits}bit_{method}"

    try:
        run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            job_type="codebook_benchmark",
            tags=[
                "codebook-benchmark",
                f"model:{model_slug}",
                f"codebook:{codebook}",
                f"bits:{bits}",
                f"method:{method}",
            ],
            config={
                **summary.get("args", {}),
                "run_id": run_id,
                "codebook": codebook,
                "bits": bits,
                "method": method,
                "model_slug": model_slug,
                "codebook_source": summary.get("codebook_source"),
                "sensitivity_mode": summary.get("sensitivity_mode"),
            },
            reinit=True,
        )
        if run is None:
            return
        if metrics:
            wandb.log(metrics)
        for key, value in row.items():
            wandb.summary[key] = value
        wandb.summary["model_path"] = summary.get("model_path")
        wandb.summary["output_dir"] = summary.get("output_dir")
        wandb.summary["run_id"] = run_id
        wandb.finish()
    except Exception as exc:
        print(f"Warning: W&B logging failed for {run_id}: {exc}")


def _format_value(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def write_reports(output_root: Path, summaries: list[dict]) -> list[dict]:
    rows = [build_result_row(summary) for summary in summaries]
    extra_columns = sorted(
        {
            column
            for summary in summaries
            for column in _flatten_lm_eval_metrics(summary)
        }
    )
    for row, summary in zip(rows, summaries):
        row.update(_flatten_lm_eval_metrics(summary))
    output_root.mkdir(parents=True, exist_ok=True)
    result_columns = RESULT_COLUMNS + [
        column for column in extra_columns if column not in RESULT_COLUMNS
    ]

    json_path = output_root / "benchmark_results.json"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    csv_path = output_root / "benchmark_results.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=result_columns)
        writer.writeheader()
        writer.writerows(rows)

    markdown_lines = [
        "| " + " | ".join(result_columns) + " |",
        "|" + "|".join(["---"] * len(result_columns)) + "|",
    ]
    for row in rows:
        markdown_lines.append(
            "| "
            + " | ".join(_format_value(row.get(column)) for column in result_columns)
            + " |"
        )
    markdown_path = output_root / "benchmark_results.md"
    markdown_path.write_text("\n".join(markdown_lines) + "\n", encoding="utf-8")

    _print_section("CODEBOOK BENCHMARK RESULTS")
    print("\t".join(result_columns))
    for row in rows:
        print("\t".join(_format_value(row.get(column)) for column in result_columns))
    print(f"\nReports: {json_path}, {csv_path}, {markdown_path}")
    return rows


def run_one(args, codebook_name: str, bits: int, method: str) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    run_id = f"{codebook_name}_{bits}bit_{method}"
    if args.run_suffix:
        run_id = f"{run_id}_{args.run_suffix}"
    output_dir = Path(args.output_root) / run_id
    summary_path = output_dir / "run_summary.json"
    started_at = time.monotonic()
    _print_section(f"RUN {run_id}")
    print(
        f"Device: {args.device} | method={method} | "
        f"codebook={codebook_name} | bits={bits} | "
        f"skip_lmhead={args.skip_lmhead}"
    )
    print(
        f"Model: {args.model_path} | output_dir={output_dir} | "
        f"seed={args.seed}"
    )
    expected_codebook_source = (
        SQUEEZELLM_DENSE_ONLY_SOURCE
        if codebook_name == "squeezellm"
        and args.squeezellm_mode == "dense-only"
        else SQUEEZELLM_HYBRID_SOURCE
        if codebook_name == "squeezellm"
        else "LeanQuant/lean_quantizer.py"
    )
    if args.resume and not args.force_eval and summary_path.exists():
        cached_summary = json.loads(summary_path.read_text(encoding="utf-8"))
        source_matches = (
            cached_summary.get("codebook_source") == expected_codebook_source
        )
        if source_matches:
            print(f"Reusing completed run: {summary_path}")
            print(f"Done. run={run_id} | resumed=True")
            return cached_summary
        print(
            "Ignoring completed run from a different codebook mode/source: "
            f"{summary_path}"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    hf_token = resolve_hf_token()
    print(f"Loading tokenizer from {args.model_path} ...")
    with tqdm(total=1, desc="Loading tokenizer", unit="step") as pbar:
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_path,
            use_fast=False,
            trust_remote_code=True,
            token=hf_token,
        )
        pbar.update(1)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    print(
        f"Loading model from {args.model_path} | "
        f"dtype={str(dtype).replace('torch.', '')} | device={args.device} ..."
    )
    def load_model():
        return AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            device_map=_device_map(args.device),
            trust_remote_code=True,
            token=hf_token,
        )

    with tqdm(total=1, desc="Loading model", unit="step") as pbar:
        model = load_model()
        pbar.update(1)
    model.eval()
    print("Model loaded and set to eval mode.")

    codebook = get_codebook(
        name=codebook_name,
        bits=bits,
        group_size=args.group_size,
        leanquant_exponent=args.leanquant_exponent,
        leanquant_percdamp=args.leanquant_percdamp,
        leanquant_act_order=args.leanquant_act_order,
        kmeans_seed=args.kmeans_seed,
    )
    cache_root = Path(args.statistics_cache_dir or Path(args.output_root) / "_statistics")
    model_slug = build_model_slug(args.model_path)
    cache_version = (
        (
            "dense_only"
            if args.squeezellm_mode == "dense-only"
            else (
                f"hybrid_range{args.squeezellm_outlier_range}"
                f"_sensitive{args.squeezellm_sensitive_percent}"
            )
        )
        if codebook_name == "squeezellm"
        else "direct_upstream"
    )
    codebook_store = CodebookStore(
        cache_root
        / "codebooks"
        / model_slug
        / f"{codebook_name}_{bits}bit_{cache_version}"
    )
    sparse_store = None
    if codebook_name == "squeezellm" and args.squeezellm_mode == "hybrid":
        sparse_store = SparseResidualStore(
            cache_root
            / "sparse_residuals"
            / model_slug
            / (
                f"squeezellm_{bits}bit_range{args.squeezellm_outlier_range}"
                f"_sensitive{args.squeezellm_sensitive_percent}"
            )
        )
    hessian_store = HessianStore(
        cache_root
        / "hessian"
        / model_slug
        / f"{codebook_name}_{bits}bit_{cache_version}"
    )

    sensitivity_path = None
    if codebook_name == "leanquant":
        if not codebook_store.complete:
            print("Preparing upstream LeanQuant calibration tokens ...")
            leanquant_samples = load_upstream_c4_tokens(
                tokenizer=tokenizer,
                n_samples=args.n_calib,
                seqlen=args.max_length,
                seed=0,
                cache_dir=cache_root / "calibration",
            )
            print("Building LeanQuant codebooks (shadow pass) ...")
            collect_leanquant_codebooks(
                model=model,
                token_samples=leanquant_samples,
                store=codebook_store,
                hessian_store=hessian_store,
                codebook=codebook,
                device=args.device,
            )
            del model, leanquant_samples
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("Reloading the original FP model after LeanQuant shadow pass ...")
            with tqdm(total=1, desc="Reloading FP model", unit="step") as pbar:
                model = load_model()
                pbar.update(1)
            model.eval()
        sensitivity_mode = "not_used"
    else:
        sensitivity_path = args.squeezellm_sensitivity
        if sensitivity_path is None:
            fisher_path = (
                cache_root
                / "fisher"
                / model_slug
                / (
                    f"c4_n{args.squeezellm_fisher_samples}"
                    f"_len{args.squeezellm_fisher_length}_seed0"
                    "_gradients_5f2a166"
                )
            )
            fisher_manifest = fisher_path / "manifest.json"
            fisher_complete = False
            if fisher_manifest.exists():
                fisher_complete = json.loads(
                    fisher_manifest.read_text(encoding="utf-8")
                ).get("complete", False)
            if not fisher_complete:
                print(
                    "Preparing Fisher data with "
                    "SqueezeLLM-gradients/datautils.py ..."
                )
                fisher_dataloader = load_squeezellm_fisher_data(
                    model_path=args.model_path,
                    num_examples=args.squeezellm_fisher_samples,
                    sequence_length=args.squeezellm_fisher_length,
                )
                collect_squeezellm_fisher(
                    model=model,
                    dataloader=fisher_dataloader,
                    output_dir=fisher_path,
                    device=args.device,
                )
                del fisher_dataloader
            sensitivity_path = str(fisher_path)
        sensitivity_mode = "fisher_checkpoint"

    sensitivity_store = SensitivityStore(sensitivity_path)
    if codebook_name == "squeezellm":
        collect_squeezellm_codebooks(
            model=model,
            sensitivity_store=sensitivity_store,
            store=codebook_store,
            sparse_store=sparse_store,
            bits=bits,
            mode=args.squeezellm_mode,
            outlier_range=args.squeezellm_outlier_range,
            sensitivity_percent=args.squeezellm_sensitive_percent,
        )
    print(
        f"Loaded quantizer: {codebook} | sensitivity={sensitivity_mode} | "
        f"codebook_cache={codebook_store.root}"
    )

    means: Dict[str, torch.Tensor] = {}
    variances: Dict[str, torch.Tensor] = {}
    calibration_texts = []
    if method in {"rbvt", "gptq"}:
        print(
            f"Loading {method.upper()} calibration data | dataset={args.calib_dataset} | "
            f"samples={args.n_calib} | max_length={args.max_length} ..."
        )
        calibration_texts = load_calibration_data(
            dataset_name=args.calib_dataset,
            tokenizer=tokenizer,
            n_samples=args.n_calib,
            seqlen=args.max_length,
            seed=args.seed,
        )
    if method == "rbvt":
        linears = [
            (name, module)
            for name, module in model.named_modules()
            if isinstance(module, nn.Linear)
            and (not args.skip_lmhead or not is_lmhead(name))
        ]
        means, variances = collect_layer_stats(
            model=model,
            tokenizer=tokenizer,
            linears=linears,
            calib_texts=calibration_texts,
            device=args.device,
            n_calib=args.n_calib,
            max_length=args.max_length,
            want_var=args.rbvt_lambda > 0.0,
        )
        print(
            f"Collected RBVT activation statistics for {len(means)}/{len(linears)} "
            f"layers | variances={len(variances)}"
        )

    _print_section(f"QUANTIZATION | {run_id}")
    if method == "gptq":
        quantization = quantize_codebook_model_gptq(
            model=model,
            tokenizer=tokenizer,
            codebook=codebook,
            codebook_store=codebook_store,
            calib_texts=calibration_texts,
            device=args.device,
            skip_lmhead=args.skip_lmhead,
            n_calib=args.n_calib,
            max_length=args.max_length,
            row_chunk=args.row_chunk,
            gptq_blocksize=args.gptq_blocksize,
            gptq_percdamp=args.gptq_percdamp,
            gptq_act_order=args.gptq_act_order,
            sparse_store=sparse_store,
        )
    else:
        quantization = quantize_with_codebook(
            model=model,
            codebook=codebook,
            method=method,
            means=means,
            variances=variances,
            skip_lmhead=args.skip_lmhead,
            row_chunk=args.row_chunk,
            rbvt_lambda=args.rbvt_lambda,
            rbvt_topk=args.rbvt_topk,
            gap_floor=args.gap_floor,
            strict_descent=args.strict_descent,
            codebook_store=codebook_store,
            sparse_store=sparse_store,
        )

    print(f"Saving to {output_dir} ...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    print("Quantized model and tokenizer saved.")
    del model, means, variances, calibration_texts
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    run_label = method.upper()
    if args.skip_perplexity:
        print("Perplexity evaluation disabled.")
        perplexity = {}
    else:
        _print_section(f"PERPLEXITY EVALUATION | {run_id}")
        perplexity = evaluate_quantized_model(
            model_path=str(output_dir),
            model_name=run_label,
            eval_device=args.device,
            eval_seed=args.seed,
            eval_stride=args.eval_stride,
            eval_max_length=args.eval_max_length,
            eval_cache_dir=args.eval_cache_dir,
            eval_samples=args.eval_samples,
            hf_token=hf_token,
        )
    if args.include_lm_eval:
        _print_section(f"LM-EVAL | {run_id}")
        print(
            f"Tasks: {', '.join(args.lm_eval_tasks)} | "
            f"batch_size={args.lm_eval_batch_size} | "
            f"num_fewshot={args.lm_eval_num_fewshot} | limit={args.lm_eval_limit}"
        )
        lm_runner = LMEvalHarnessRunner(
            tasks=args.lm_eval_tasks,
            device=args.device,
            batch_size=args.lm_eval_batch_size,
            num_fewshot=args.lm_eval_num_fewshot,
            limit=args.lm_eval_limit,
            output_dir=args.lm_eval_output_dir,
            run_name=f"{datetime.now():%Y%m%d-%H%M%S}_{run_id}",
            hf_token=hf_token,
        )
        lm_eval = lm_runner.run({run_label: str(output_dir)})
    else:
        print("lm-eval disabled.")
        lm_eval = {}

    summary = {
        "model_path": args.model_path,
        "output_dir": str(output_dir),
        "run_label": run_label,
        "codebook": codebook_name,
        "bits": bits,
        "method": method,
        "sensitivity_mode": sensitivity_mode,
        "codebook_source": expected_codebook_source,
        "quantization": quantization,
        "calibration": {
            "dataset": args.calib_dataset,
            "n_calib": args.n_calib,
            "max_length": args.max_length,
            "seed": args.seed,
        },
        "evaluation": {
            "perplexity": perplexity,
            "lm_eval": lm_eval,
            "tasks": args.lm_eval_tasks if args.include_lm_eval else [],
        },
        "args": vars(args),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(f"Saved run summary to {summary_path}")
    elapsed = time.monotonic() - started_at
    if args.use_wandb:
        log_summary_to_wandb(args, summary, run_id, elapsed)
    if not args.keep_model:
        cleanup_output_dir(str(output_dir))
    print(f"Done. run={run_id} | elapsed={elapsed:.1f}s")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="LeanQuant/SqueezeLLM codebook benchmark for RTN, RBVT, and GPTQ"
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", default="./outputs/codebook_benchmark")
    parser.add_argument(
        "--codebooks",
        nargs="+",
        choices=["leanquant", "squeezellm"],
        default=["leanquant", "squeezellm"],
    )
    parser.add_argument("--bits", nargs="+", type=int, choices=[3, 4], default=[3, 4])
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=["rtn", "rbvt", "gptq"],
        default=["rtn", "rbvt"],
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--force-eval",
        action="store_true",
        help="Ignore existing run_summary.json and recompute quantization/evaluation; codebook/statistics caches are still reused.",
    )
    parser.add_argument("--keep-model", action="store_true")
    parser.add_argument("--run-suffix", default="")

    parser.add_argument("--skip-lmhead", action="store_true", default=True)
    parser.add_argument("--no-skip-lmhead", dest="skip_lmhead", action="store_false")
    parser.add_argument("--calib-dataset", choices=["c4", "wikitext2"], default="c4")
    parser.add_argument("--n-calib", type=int, default=128)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--group-size", type=int, default=-1)
    parser.add_argument(
        "--row-chunk",
        type=int,
        default=1024,
        help="General quantization/RBVT row chunk; matches main.py by default",
    )
    parser.add_argument("--gptq-blocksize", type=int, default=128)
    parser.add_argument("--gptq-percdamp", type=float, default=0.01)
    parser.add_argument(
        "--gptq-act-order",
        dest="gptq_act_order",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--no-gptq-act-order",
        dest="gptq_act_order",
        action="store_false",
    )
    parser.add_argument("--leanquant-exponent", type=float, default=4.0)
    parser.add_argument("--leanquant-percdamp", type=float, default=0.1)
    parser.add_argument("--kmeans-seed", type=int, default=0)
    parser.add_argument(
        "--leanquant-act-order",
        dest="leanquant_act_order",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-leanquant-act-order",
        dest="leanquant_act_order",
        action="store_false",
    )
    parser.add_argument(
        "--squeezellm-sensitivity",
        default=None,
        help="Existing Fisher checkpoint; omitted means collect it upstream-style",
    )
    parser.add_argument("--squeezellm-fisher-samples", type=int, default=100)
    parser.add_argument("--squeezellm-fisher-length", type=int, default=512)
    parser.add_argument(
        "--squeezellm-mode",
        choices=["dense-only", "hybrid"],
        default="hybrid",
    )
    parser.add_argument("--squeezellm-outlier-range", type=float, default=1.8)
    parser.add_argument("--squeezellm-sensitive-percent", type=float, default=0.05)
    parser.add_argument("--statistics-cache-dir", default=None)

    parser.add_argument("--rbvt-lambda", type=float, default=1.0)
    parser.add_argument("--rbvt-topk", type=int, default=0)
    parser.add_argument("--gap-floor", type=float, default=1e-8)
    parser.add_argument("--strict-descent", action="store_true", default=True)
    parser.add_argument(
        "--allow-overshoot",
        dest="strict_descent",
        action="store_false",
    )

    parser.add_argument("--eval-stride", type=int, default=512)
    parser.add_argument("--eval-max-length", type=int, default=2048)
    parser.add_argument("--eval-samples", type=int, default=2000)
    parser.add_argument("--eval-cache-dir", default="./dataset_cache")
    parser.add_argument("--skip-perplexity", action="store_true")
    parser.add_argument("--include-lm-eval", action="store_true", default=True)
    parser.add_argument("--no-lm-eval", dest="include_lm_eval", action="store_false")
    parser.add_argument("--lm-eval-tasks", nargs="+", default=list(DEFAULT_LM_EVAL_TASKS))
    parser.add_argument("--lm-eval-batch-size", default="auto")
    parser.add_argument("--lm-eval-num-fewshot", type=int, default=None)
    parser.add_argument("--lm-eval-limit", type=float, default=None)
    parser.add_argument("--lm-eval-output-dir", default="./outputs/lm_eval_codebooks")
    parser.add_argument(
        "--use-wandb",
        dest="use_wandb",
        action="store_true",
        default=os.getenv("USE_WANDB", "0") == "1",
    )
    parser.add_argument("--no-wandb", dest="use_wandb", action="store_false")
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "rbvtquant"))
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY") or None)
    return parser


def main():
    load_runtime_env()
    args = build_parser().parse_args()
    if "leanquant" in args.codebooks:
        load_leanquant_upstream()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            f"{args.device} was requested, but CUDA is unavailable. "
            "Llama-3.1-8B full evaluation requires a CUDA machine."
        )
    if args.rbvt_lambda < 0.0:
        raise ValueError("--rbvt-lambda must be non-negative")
    if args.squeezellm_outlier_range <= 0.0:
        raise ValueError("--squeezellm-outlier-range must be positive")
    if not 0.0 < args.squeezellm_sensitive_percent < 100.0:
        raise ValueError("--squeezellm-sensitive-percent must be between 0 and 100")
    if args.group_size != -1:
        raise ValueError(
            "Exact upstream LeanQuant/SqueezeLLM codebooks require --group-size=-1"
        )
    if not args.skip_lmhead:
        raise ValueError(
            "LeanQuant and SqueezeLLM upstream flows do not quantize lm_head; "
            "--no-skip-lmhead is incompatible with exact mode"
        )

    _set_seed(args.seed)
    total_runs = len(args.codebooks) * len(args.bits) * len(args.methods)
    _print_section("RBVTQUANT CODEBOOK BENCHMARK")
    print(
        f"Model: {args.model_path} | Device: {args.device} | "
        f"Runs: {total_runs} | seed={args.seed}"
    )
    print(
        f"Codebooks: {', '.join(args.codebooks)} | "
        f"bits={args.bits} | methods={args.methods}"
    )
    print(
        f"Calibration: dataset={args.calib_dataset}, samples={args.n_calib}, "
        f"max_length={args.max_length}"
    )
    print(
        "Upstream codebook statistics: "
        f"LeanQuant=C4/{args.n_calib}x{args.max_length}, seed=0, "
        "true-sequential full Hessian; "
        f"SqueezeLLM=C4/{args.squeezellm_fisher_samples}x"
        f"{args.squeezellm_fisher_length}, Fisher seed=0, "
        f"mode={args.squeezellm_mode}"
        + (
            f", outlier_range={args.squeezellm_outlier_range}, "
            f"sensitive={args.squeezellm_sensitive_percent}%"
            if args.squeezellm_mode == "hybrid"
            else ""
        )
    )
    print(
        f"Evaluation: stride={args.eval_stride}, "
        f"max_length={args.eval_max_length}, samples={args.eval_samples}, "
        f"perplexity={not args.skip_perplexity}, "
        f"lm_eval={args.include_lm_eval}, "
        f"lm_eval_tasks={args.lm_eval_tasks}"
    )
    print(f"W&B logging: {args.use_wandb} | project={args.wandb_project}")

    benchmark_started_at = time.monotonic()
    summaries = []
    run_index = 0
    for codebook_name in args.codebooks:
        for bits in args.bits:
            for method in args.methods:
                run_index += 1
                print(
                    f"\nStarting run {run_index}/{total_runs}: "
                    f"{codebook_name} {bits}-bit {method.upper()}"
                )
                summaries.append(run_one(args, codebook_name, bits, method))
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    write_reports(Path(args.output_root), summaries)
    elapsed = time.monotonic() - benchmark_started_at
    print(f"Done. completed_runs={len(summaries)}/{total_runs} | elapsed={elapsed:.1f}s")


if __name__ == "__main__":
    main()
