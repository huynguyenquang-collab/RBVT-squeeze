"""Fisher collection plus direct invocation of SqueezeLLM's upstream LUT code."""

from __future__ import annotations

import json
import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from .codebook_store import CodebookStore
from .sensitivity_store import SensitivityStore
from .sparse_residual_store import SparseResidualStore
from .upstream_imports import (
    SQUEEZELLM_GRADIENTS_SOURCE,
    load_squeezellm_gradients,
    load_squeezellm_kmeans,
    load_squeezellm_model_parse,
    load_squeezellm_remove_outliers,
)
from calibration_utils import get_redpajama_calibration_data


def _env_flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() in {"1", "true", "yes", "on"}


def _squeezellm_linears(model) -> list[tuple[str, nn.Linear]]:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("SqueezeLLM Fisher collection currently requires a Llama model")
    model_parse = load_squeezellm_model_parse()
    names = model_parse.get_sequential("llama")
    _, get_modules, _ = load_squeezellm_gradients()
    result = []
    for layer_index, layer in enumerate(model.model.layers):
        modules = get_modules(layer)
        if len(names) != len(modules):
            raise RuntimeError(
                "SqueezeLLM and SqueezeLLM-gradients disagree on Llama layer count: "
                f"{len(names)} names != {len(modules)} modules"
            )
        for relative_name, module in zip(names, modules):
            if not isinstance(module, nn.Linear):
                raise TypeError(
                    f"Expected Linear at model.layers.{layer_index}.{relative_name}"
                )
            result.append((f"model.layers.{layer_index}.{relative_name}", module))
    return result


def _squeezellm_layer_modules(model) -> list[list[tuple[str, nn.Linear]]]:
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("SqueezeLLM Fisher collection currently requires a Llama-style model")
    model_parse = load_squeezellm_model_parse()
    names = model_parse.get_sequential("llama")
    _, get_modules, _ = load_squeezellm_gradients()
    result = []
    for layer_index, layer in enumerate(model.model.layers):
        modules = get_modules(layer)
        if len(names) != len(modules):
            raise RuntimeError(
                "SqueezeLLM and SqueezeLLM-gradients disagree on Llama layer count: "
                f"{len(names)} names != {len(modules)} modules"
            )
        layer_modules = []
        for relative_name, module in zip(names, modules):
            if not isinstance(module, nn.Linear):
                raise TypeError(
                    f"Expected Linear at model.layers.{layer_index}.{relative_name}"
                )
            layer_modules.append((f"model.layers.{layer_index}.{relative_name}", module))
        result.append(layer_modules)
    return result


def _layer_groups(layer_count: int, layers_per_pass: int) -> list[list[int]]:
    if layers_per_pass <= 0:
        return [list(range(layer_count))]
    return [
        list(range(start, min(start + layers_per_pass, layer_count)))
        for start in range(0, layer_count, layers_per_pass)
    ]


def load_squeezellm_fisher_data(
    model_path: str,
    tokenizer=None,
    dataset_name: str = "c4",
    num_examples: int = 100,
    sequence_length: int = 512,
    seed: int = 0,
    cache_dir: str | Path = "./calibration_cache",
):
    """Load Fisher calibration tokens for SqueezeLLM sensitivity collection."""

    dataset_name = dataset_name.lower()
    if dataset_name in {"redpajama", "red_pajama"}:
        if tokenizer is None:
            raise ValueError("tokenizer is required for RedPajama SqueezeLLM Fisher calibration")
        tokens = get_redpajama_calibration_data(
            tokenizer=tokenizer,
            n_samples=num_examples,
            seqlen=sequence_length,
            seed=seed,
            return_tensors=True,
            cache_dir=cache_dir,
        )
        return [(item,) for item in tokens]
    if dataset_name != "c4":
        raise ValueError(f"Unsupported SqueezeLLM Fisher dataset: {dataset_name}")

    # Preserve the original upstream C4 loader path by default.
    get_loaders, _, _ = load_squeezellm_gradients()
    dataloader, _ = get_loaders(
        "c4",
        model=model_path,
        seqlen=sequence_length,
        nsamples=num_examples,
    )
    return dataloader


def collect_squeezellm_fisher(
    model,
    dataloader,
    output_dir: str | Path,
    device: str,
) -> Path:
    """Run the Fisher operations defined by SqueezeLLM-gradients/run.py."""

    output_dir = Path(output_dir)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("complete", False):
            print(f"Reusing SqueezeLLM Fisher cache: {output_dir}")
            return output_dir

    output_dir.mkdir(parents=True, exist_ok=True)
    layer_modules = _squeezellm_layer_modules(model)
    layers_per_pass = int(os.environ.get("SQUEEZELLM_FISHER_LAYERS_PER_PASS", "1"))
    _, _, square_grad_hook = load_squeezellm_gradients()
    accum_device = os.environ.get("SQUEEZELLM_FISHER_ACCUM_DEVICE", "gpu").lower()
    if accum_device not in {"gpu", "cpu"}:
        raise ValueError("SQUEEZELLM_FISHER_ACCUM_DEVICE must be 'gpu' or 'cpu'")
    grad_checkpointing = _env_flag("SQUEEZELLM_FISHER_GRAD_CHECKPOINTING", "0")
    if hasattr(model, "config"):
        model.config.use_cache = False
    if grad_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    print(
        "SqueezeLLM Fisher accumulation: "
        f"{accum_device} | gradient_checkpointing={grad_checkpointing} | "
        f"layers_per_pass={layers_per_pass}"
    )
    for param in model.parameters():
        param.requires_grad_(False)
    model.train()

    layers = {}
    try:
        for group in _layer_groups(len(layer_modules), layers_per_pass):
            pending = []
            for layer_idx in group:
                expected = [
                    output_dir / f"{layer_name.replace('.', '_')}.pt"
                    for layer_name, _ in layer_modules[layer_idx]
                ]
                if all(path.exists() for path in expected):
                    print(f"Skipping existing Fisher layer {layer_idx}")
                    continue
                pending.append(layer_idx)
            if not pending:
                continue

            selected = []
            handles = []
            cpu_accum = {}
            try:
                for layer_idx in pending:
                    for layer_name, module in layer_modules[layer_idx]:
                        module.weight.requires_grad_(True)
                        module.weight.grad = None
                        selected.append((layer_name, module))
                        if accum_device == "cpu":
                            cpu_accum[layer_name] = torch.zeros_like(
                                module.weight,
                                dtype=torch.float32,
                                device="cpu",
                            )

                            def make_cpu_hook(target_name):
                                def hook(grad):
                                    cpu_accum[target_name].add_(grad.detach().float().cpu().square())
                                    return torch.zeros_like(grad)

                                return hook

                            handles.append(module.weight.register_hook(make_cpu_hook(layer_name)))
                        else:
                            handles.append(module.weight.register_hook(square_grad_hook))

                group_label = (
                    str(pending[0])
                    if len(pending) == 1
                    else f"{pending[0]}-{pending[-1]}"
                )
                model.zero_grad(set_to_none=True)
                for step, data in enumerate(
                    tqdm(dataloader, desc=f"Collecting SqueezeLLM Fisher layers {group_label}"),
                    start=1,
                ):
                    input_ids = data[0].to(device)
                    outputs = model(input_ids=input_ids, labels=input_ids, use_cache=False)
                    outputs.loss.backward()
                    if accum_device == "cpu":
                        for _, module in selected:
                            module.weight.grad = None
                        model.zero_grad(set_to_none=True)
                    del input_ids, outputs
                    if torch.cuda.is_available() and step % 32 == 0:
                        torch.cuda.empty_cache()

                for layer_name, module in selected:
                    if accum_device == "cpu":
                        fisher = cpu_accum[layer_name]
                    else:
                        if module.weight.grad is None:
                            raise RuntimeError(f"No Fisher gradient collected for {layer_name}")
                        fisher = module.weight.grad.detach().float().cpu()
                    filename = layer_name.replace(".", "_") + ".pt"
                    torch.save(fisher, output_dir / filename)
                    layers[f"{layer_name}.weight"] = filename
            finally:
                for handle in handles:
                    handle.remove()
                for _, module in selected:
                    module.weight.requires_grad_(False)
                    module.weight.grad = None
                model.zero_grad(set_to_none=True)
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        if grad_checkpointing and hasattr(model, "gradient_checkpointing_disable"):
            model.gradient_checkpointing_disable()

    for layer in layer_modules:
        for layer_name, _ in layer:
            filename = layer_name.replace(".", "_") + ".pt"
            layers.setdefault(f"{layer_name}.weight", filename)
    model.zero_grad(set_to_none=True)

    manifest_path.write_text(
        json.dumps(
            {
                "complete": True,
                "format": "sum_of_per_sample_gradient_squares",
                "algorithm": "squeezellm_gradients_upstream",
                "source": SQUEEZELLM_GRADIENTS_SOURCE,
                "layers": layers,
                "num_examples": len(dataloader),
                "sequence_length": int(dataloader[0][0].shape[1]),
                "seed": 0,
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    model.eval()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return output_dir


def collect_squeezellm_codebooks(
    model,
    sensitivity_store: SensitivityStore,
    store: CodebookStore,
    sparse_store: SparseResidualStore | None,
    bits: int,
    mode: str = "hybrid",
    outlier_range: float = 1.8,
    sensitivity_percent: float = 0.05,
):
    """Generate upstream dense-only or dense+sparse+sensitive SqueezeLLM LUTs."""

    if mode not in {"dense-only", "hybrid"}:
        raise ValueError(f"Unsupported SqueezeLLM mode: {mode}")
    if mode == "hybrid" and sparse_store is None:
        raise ValueError("SqueezeLLM hybrid mode requires a sparse residual store")
    metadata = {
        "algorithm": (
            "squeezellm_upstream_dense_sparse_sensitive"
            if mode == "hybrid"
            else "squeezellm_upstream_dense_only"
        ),
        "source": (
            SQUEEZELLM_GRADIENTS_SOURCE
            + " + SqueezeLLM/quantization/nuq.py"
        ),
        "bits": bits,
        "fisher": str(sensitivity_store.path),
        "mode": mode,
    }
    if mode == "hybrid":
        metadata.update(
            {
                "outlier_range": outlier_range,
                "sensitivity_percent": sensitivity_percent,
            }
        )
    stores_complete = store.complete and (
        mode == "dense-only" or sparse_store.complete
    )
    if stores_complete:
        store.validate(metadata)
        if sparse_store is not None:
            sparse_store.validate(metadata)
        print(f"Reusing SqueezeLLM {mode} cache: {store.root}")
        return
    store.initialize(metadata)
    if sparse_store is not None:
        sparse_store.initialize(metadata)

    kmeans_fit = load_squeezellm_kmeans()
    remove_outliers = (
        load_squeezellm_remove_outliers()
        if mode == "hybrid"
        else None
    )
    model_parse = load_squeezellm_model_parse()
    relative_names = model_parse.get_sequential("llama")
    short_names = model_parse.get_module_names("llama")
    workers = os.cpu_count() or 1
    mode_details = (
        f"outlier_range={outlier_range} | sensitivity={sensitivity_percent}% | "
        if mode == "hybrid"
        else ""
    )
    print(f"Building SqueezeLLM {mode} LUTs | {mode_details}workers={workers}")
    with Pool(workers) as pool:
        for layer_index, layer in enumerate(
            tqdm(model.model.layers, desc=f"SqueezeLLM {mode} LUTs")
        ):
            model_layer = {}
            gradient_layer = {}
            outlier_config = {}
            full_names = {}
            for short_name, relative_name in zip(short_names, relative_names):
                module = layer.get_submodule(relative_name)
                full_name = f"model.layers.{layer_index}.{relative_name}"
                weight = module.weight.detach().float().cpu()
                gradient = sensitivity_store.get(full_name, weight.shape)
                model_layer[short_name] = weight
                gradient_layer[short_name] = gradient
                if mode == "hybrid":
                    values = weight.numpy()
                    q1 = np.quantile(values, 0.25)
                    q3 = np.quantile(values, 0.75)
                    outlier_config[short_name] = max(
                        abs(q1 - outlier_range * (q3 - q1)),
                        abs(q3 + outlier_range * (q3 - q1)),
                    )
                full_names[short_name] = full_name

            sparse_lists = None
            if mode == "hybrid":
                sparse_lists = remove_outliers(
                    model=model_layer,
                    sensitivity=sensitivity_percent,
                    outlier_config=outlier_config,
                    gradients=gradient_layer,
                )[0]

            for sparse_index, short_name in enumerate(short_names):
                full_name = full_names[short_name]
                dense_weight = model_layer[short_name]
                gradient = gradient_layer[short_name]
                if sparse_store is not None:
                    sparse_store.put(full_name, sparse_lists[sparse_index])
                tasks = []
                for row, gradient_row in zip(
                    dense_weight.numpy(),
                    gradient.numpy(),
                ):
                    sample_weight = gradient_row * (row != 0)
                    if np.sum(sample_weight) == 0:
                        sample_weight = np.ones_like(sample_weight)
                    tasks.append((row.reshape(-1, 1), sample_weight, 2**bits))
                results = list(pool.imap(kmeans_fit, tasks))
                centers = torch.from_numpy(
                    np.stack([result[0] for result in results])
                ).float()
                store.put(full_name, torch.sort(centers, dim=1).values.unsqueeze(1))

    store.mark_complete()
    if sparse_store is not None:
        sparse_store.mark_complete()


__all__ = [
    "collect_squeezellm_codebooks",
    "collect_squeezellm_fisher",
    "load_squeezellm_fisher_data",
]
