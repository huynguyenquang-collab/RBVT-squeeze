"""Layer-sequential collection using LeanQuant's upstream implementation."""

from __future__ import annotations

from types import SimpleNamespace

import torch
import torch.nn as nn
from tqdm import tqdm

from .base_codebook import BaseCodebook
from .codebook_store import CodebookStore
from .hessian_store import HessianStore
from .upstream_imports import load_leanquant_upstream


class _StopCapture(Exception):
    pass


class _InputCatcher(nn.Module):
    def __init__(self, module: nn.Module, inputs: list[torch.Tensor], state: dict):
        super().__init__()
        self.module = module
        self.inputs = inputs
        self.state = state

    def forward(self, hidden_states, **kwargs):
        self.inputs.append(hidden_states.detach())
        self.state["kwargs"] = kwargs
        raise _StopCapture


def _forward_layer(layer: nn.Module, hidden_states: torch.Tensor, kwargs: dict):
    output = layer(hidden_states, **kwargs)
    return output[0] if isinstance(output, (tuple, list)) else output


def _get_subsets(layer: nn.Module) -> list[list[str]]:
    groups = [
        ["self_attn.k_proj", "self_attn.v_proj", "self_attn.q_proj"],
        ["self_attn.o_proj"],
        ["mlp.up_proj", "mlp.gate_proj"],
        ["mlp.down_proj"],
    ]
    for names in groups:
        for name in names:
            module = layer.get_submodule(name)
            if not isinstance(module, nn.Linear):
                raise TypeError(f"LeanQuant expected Linear module {name!r}")
    return groups


@torch.no_grad()
def collect_leanquant_codebooks(
    model,
    token_samples: list[torch.Tensor],
    store: CodebookStore,
    hessian_store: HessianStore,
    codebook: BaseCodebook,
    device: str,
):
    """Run upstream LeanQuant true-sequential quantization and persist its grids."""

    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise TypeError("LeanQuant collection currently requires a Llama model")
    LeanQuant, Quantizer = load_leanquant_upstream()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    metadata = {
        "algorithm": "leanquant_direct_upstream_true_sequential",
        "source": "LeanQuant/lean_quantizer.py",
        "bits": codebook.bits,
        "exponent": codebook.exponent,
        "percdamp": codebook.percdamp,
        "act_order": codebook.act_order,
        "kmeans_seed": codebook.kmeans_seed,
        "block_size": codebook.gptq_block_size,
        "num_examples": len(token_samples),
        "sequence_length": int(token_samples[0].shape[1]),
    }
    upstream_args = SimpleNamespace(
        offload_threshold=53248,
        exponent=float(codebook.exponent),
        wbits=codebook.bits,
        kmeans_seed=codebook.kmeans_seed,
        save_path="rbvtquant-upstream-codebook",
    )
    if store.complete:
        if store.metadata != metadata:
            raise ValueError(
                f"LeanQuant cache metadata mismatch: {store.metadata} != {metadata}"
            )
        print(f"Reusing LeanQuant codebook cache: {store.root}")
    if hessian_store.complete:
        if hessian_store.metadata != metadata:
            raise ValueError(
                f"LeanQuant Hessian cache metadata mismatch: {hessian_store.metadata} != {metadata}"
            )
        print(f"Reusing LeanQuant Hessian cache: {hessian_store.root}")
    if store.complete and hessian_store.complete:
        return
    if not store.complete:
        store.initialize(metadata)
    if not hessian_store.complete:
        hessian_store.initialize(metadata)

    layers = model.model.layers
    original_use_cache = getattr(model.config, "use_cache", None)
    if original_use_cache is not None:
        model.config.use_cache = False
    model.eval()
    print(
        f"Starting LeanQuant shadow pass | layers={len(layers)} | "
        f"samples={len(token_samples)}"
    )

    captured_inputs: list[torch.Tensor] = []
    capture_state: dict = {}
    first_layer = layers[0]
    layers[0] = _InputCatcher(first_layer, captured_inputs, capture_state)
    try:
        for input_ids in tqdm(token_samples, desc="Capturing LeanQuant inputs"):
            try:
                model(input_ids=input_ids.to(device), use_cache=False)
            except _StopCapture:
                pass
    finally:
        layers[0] = first_layer

    if len(captured_inputs) != len(token_samples):
        raise RuntimeError(
            f"Captured {len(captured_inputs)} LeanQuant inputs for "
            f"{len(token_samples)} samples"
        )
    print("Captured LeanQuant inputs; starting layer-wise Hessian/codebook collection ...")
    layer_kwargs = capture_state.get("kwargs", {})
    inps = captured_inputs

    for layer_index, layer in enumerate(
        tqdm(layers, desc="LeanQuant layers", unit="layer")
    ):
        print(f"LeanQuant shadow layer {layer_index + 1}/{len(layers)}")
        subsets = _get_subsets(layer)
        outs: list[torch.Tensor] = []

        for names in tqdm(
            subsets,
            desc=f"Layer {layer_index + 1}: subsets",
            unit="subset",
            leave=False,
        ):
            workers = {}
            handles = []
            for name in names:
                module = layer.get_submodule(name)
                cache_key = f"model.layers.{layer_index}.{name}"
                leanquant = LeanQuant(module)
                workers[name] = (module, cache_key, leanquant)
                hessian = hessian_store.get(cache_key)
                if hessian is None:
                    handles.append(
                        module.register_forward_hook(
                            lambda _module, inputs, output, target=leanquant: target.add_batch(
                                inputs[0].data,
                                output.data,
                            )
                        )
                    )
                else:
                    leanquant.H = hessian.to(module.weight.device)

            if handles:
                try:
                    for sample in tqdm(
                        inps,
                        desc=f"Layer {layer_index + 1}: Hessian samples",
                        unit="sample",
                        leave=False,
                    ):
                        _forward_layer(layer, sample, layer_kwargs)
                finally:
                    for handle in handles:
                        handle.remove()
                for _name, (_module, cache_key, leanquant) in workers.items():
                    if not hessian_store.has(cache_key):
                        hessian_store.put(cache_key, leanquant.H)

            for name in names:
                module, cache_key, leanquant = workers[name]
                leanquant.quantizer = Quantizer()
                leanquant.quantizer.configure(
                    codebook.bits,
                    perchannel=True,
                    sym=False,
                    mse=False,
                )
                leanquant.fasterquant(
                    blocksize=codebook.gptq_block_size,
                    percdamp=codebook.percdamp,
                    groupsize=-1,
                    actorder=codebook.act_order,
                    static_groups=False,
                    args=upstream_args,
                )
                centers = torch.sort(
                    leanquant.quant_grid.detach().float(),
                    dim=1,
                ).values
                store.put(cache_key, centers.unsqueeze(1))
                leanquant.free()
                del centers, leanquant
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        for sample in tqdm(
            inps,
            desc=f"Layer {layer_index + 1}: propagate outputs",
            unit="sample",
            leave=False,
        ):
            outs.append(_forward_layer(layer, sample, layer_kwargs).detach())
        inps = outs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    store.mark_complete()
    hessian_store.mark_complete()
    if original_use_cache is not None:
        model.config.use_cache = original_use_cache


__all__ = ["collect_leanquant_codebooks"]
