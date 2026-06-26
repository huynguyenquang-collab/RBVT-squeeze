"""
Non-uniform GPTQ adapters for RBVT-squeeze.

This keeps the Hessian-aware sequential error compensation from GPTQ, but uses
nearest-codeword projection on an existing non-uniform codebook. The codebook
can come from the built-in quantizers or from cached upstream LeanQuant /
SqueezeLLM grids.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
from tqdm import tqdm

from quantizers.base_codebook import CodebookContext
from quantizers.base_quantizer import QuantResult


@dataclass
class GPTQStats:
    method: str
    num_linear_layers: int
    skip_lmhead: bool
    total_error: float
    total_time_sec: float
    gptq_blocksize: int
    gptq_percdamp: float
    gptq_act_order: bool


def is_lmhead(name: str) -> bool:
    return "lm_head" in name.lower() or name.endswith("lm_head")


@torch.no_grad()
def _realized_codebooks(qres: QuantResult, device: torch.device) -> torch.Tensor:
    if qres.block_codebooks is not None:
        return qres.block_codebooks.to(device).float()
    q = qres.q_levels.to(device).float()
    scales = qres.block_scales.to(device).float()
    if qres.block_zeros is not None:
        zeros = qres.block_zeros.to(device).float()
        return scales.unsqueeze(-1) * (q.view(1, 1, -1) - zeros.unsqueeze(-1))
    return scales.unsqueeze(-1) * q.view(1, 1, -1)


@torch.no_grad()
def _nearest_codeword(w: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    idx = (w.unsqueeze(1) - levels).abs().argmin(dim=1)
    row_ids = torch.arange(w.shape[0], device=w.device)
    return levels[row_ids, idx]


class NonUniformGPTQ:
    def __init__(self, layer: nn.Linear, qres: QuantResult, reference_weight: torch.Tensor | None = None):
        self.layer = layer
        self.dev = self.layer.weight.device
        W = (reference_weight if reference_weight is not None else layer.weight.data).clone().float()
        self.reference_weight = W
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        self.H = torch.zeros((self.columns, self.columns), device=self.dev)
        self.nsamples = 0
        self.qres = qres
        self.codebooks = _realized_codebooks(qres=qres, device=self.dev)
        self.block_size = qres.block_size

    def add_batch(self, inp, _out):
        if len(inp.shape) == 2:
            inp = inp.unsqueeze(0)
        tmp = inp.shape[0]
        if len(inp.shape) == 3:
            inp = inp.reshape((-1, inp.shape[-1]))
        inp = inp.t()
        self.H *= self.nsamples / (self.nsamples + tmp)
        self.nsamples += tmp
        inp = math.sqrt(2 / self.nsamples) * inp.float()
        self.H += inp.matmul(inp.t())

    @torch.no_grad()
    def fasterquant(
        self,
        blocksize: int = 128,
        percdamp: float = 0.01,
        actorder: bool = False,
        freeze_mask: torch.Tensor | None = None,
    ) -> tuple[float, float]:
        W = self.reference_weight.clone()
        tick = time.time()

        H = self.H
        del self.H
        dead = torch.diag(H) == 0
        H[dead, dead] = 1
        W[:, dead] = 0

        if freeze_mask is not None:
            freeze_mask = freeze_mask.to(device=self.dev, dtype=torch.bool)
            if freeze_mask.shape != W.shape:
                raise ValueError(
                    f"freeze_mask has shape {tuple(freeze_mask.shape)}, expected {tuple(W.shape)}"
                )

        if actorder:
            perm = torch.argsort(torch.diag(H), descending=True)
            W = W[:, perm]
            H = H[perm][:, perm]
            freeze_mask = freeze_mask[:, perm] if freeze_mask is not None else None
            invperm = torch.argsort(perm)
        else:
            perm = None
            invperm = None

        losses = torch.zeros_like(W)
        Q = torch.zeros_like(W)

        damp = percdamp * torch.mean(torch.diag(H))
        diag = torch.arange(self.columns, device=self.dev)
        H[diag, diag] += damp
        H = torch.linalg.cholesky(H)
        H = torch.cholesky_inverse(H)
        H = torch.linalg.cholesky(H, upper=True)
        Hinv = H

        for i1 in range(0, self.columns, blocksize):
            i2 = min(i1 + blocksize, self.columns)
            count = i2 - i1

            W1 = W[:, i1:i2].clone()
            Q1 = torch.zeros_like(W1)
            Err1 = torch.zeros_like(W1)
            Losses1 = torch.zeros_like(W1)
            Hinv1 = Hinv[i1:i2, i1:i2]

            for i in range(count):
                global_col = i1 + i
                orig_col = int(perm[global_col].item()) if perm is not None else global_col
                block_idx = orig_col // self.block_size

                w = W1[:, i]
                d = Hinv1[i, i]
                levels = self.codebooks[:, block_idx, :]
                q = _nearest_codeword(w=w, levels=levels)
                if freeze_mask is not None:
                    frozen = freeze_mask[:, global_col]
                    q = torch.where(frozen, w, q)
                Q1[:, i] = q
                Losses1[:, i] = (w - q) ** 2 / d**2

                err1 = (w - q) / d
                W1[:, i:] -= err1.unsqueeze(1).matmul(Hinv1[i, i:].unsqueeze(0))
                Err1[:, i] = err1

            Q[:, i1:i2] = Q1
            losses[:, i1:i2] = Losses1 / 2
            W[:, i2:] -= Err1.matmul(Hinv[i1:i2, i2:])

        if actorder and invperm is not None:
            Q = Q[:, invperm]

        self.layer.weight.data = Q.reshape(self.layer.weight.shape).to(self.layer.weight.data.dtype)
        elapsed = time.time() - tick
        return float(torch.sum(losses).item()), elapsed

    def free(self):
        self.H = None
        torch.cuda.empty_cache()


def _linear_layers(model, skip_lmhead: bool) -> List[Tuple[str, nn.Linear]]:
    linears = [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]
    if skip_lmhead:
        linears = [(name, module) for name, module in linears if not is_lmhead(name)]
    return linears


@torch.no_grad()
def quantize_model_gptq(
    model,
    tokenizer,
    quantizer,
    calib_texts: List[str],
    device: str,
    skip_lmhead: bool = True,
    n_calib: int = 128,
    max_length: int = 512,
    row_chunk: int = 1024,
    gptq_blocksize: int = 128,
    gptq_percdamp: float = 0.01,
    gptq_act_order: bool = False,
):
    linears = _linear_layers(model, skip_lmhead=skip_lmhead)
    print(
        f"Quantizing {len(linears)} Linear layers "
        f"({'skipping' if skip_lmhead else 'including'} lm_head) | method=gptq"
    )

    total_error = 0.0
    total_time_sec = 0.0

    for name, module in tqdm(linears, desc="Quantizing layers"):
        weight = module.weight.data
        qres = quantizer.quantize(weight, row_chunk=row_chunk)
        gptq = NonUniformGPTQ(layer=module, qres=qres, reference_weight=weight)

        def add_batch(_m, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            gptq.add_batch(x.data, out.data)

        handle = module.register_forward_hook(add_batch)
        try:
            for i, text in enumerate(calib_texts[:n_calib]):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                model(**inputs, use_cache=False)
                if (i + 1) % 16 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            handle.remove()

        err, elapsed = gptq.fasterquant(
            blocksize=gptq_blocksize,
            percdamp=gptq_percdamp,
            actorder=gptq_act_order,
        )
        gptq.free()
        total_error += err
        total_time_sec += elapsed
        print(f"GPTQ layer done | {name} | error={err:.6e} | time={elapsed:.2f}s")

    print(f"GPTQ summary | total_error={total_error:.6e} | total_time_sec={total_time_sec:.2f}")
    return model, GPTQStats(
        method="gptq",
        num_linear_layers=len(linears),
        skip_lmhead=skip_lmhead,
        total_error=total_error,
        total_time_sec=total_time_sec,
        gptq_blocksize=gptq_blocksize,
        gptq_percdamp=gptq_percdamp,
        gptq_act_order=gptq_act_order,
    )


@torch.no_grad()
def quantize_codebook_model_gptq(
    model,
    tokenizer,
    codebook,
    codebook_store,
    calib_texts: List[str],
    device: str,
    skip_lmhead: bool = True,
    n_calib: int = 128,
    max_length: int = 512,
    row_chunk: int = 1024,
    gptq_blocksize: int = 128,
    gptq_percdamp: float = 0.01,
    gptq_act_order: bool = False,
    sparse_store=None,
):
    linears = _linear_layers(model, skip_lmhead=skip_lmhead)
    print(
        f"Quantizing {len(linears)} Linear layers "
        f"({'skipping' if skip_lmhead else 'including'} lm_head) | method=gptq"
    )

    total_error = 0.0
    total_time_sec = 0.0
    total_sparse_values = 0

    for name, module in tqdm(linears, desc="Quantizing layers"):
        weight = module.weight.data
        sparse_residual = None
        sparse_mask = None
        dense_weight = weight
        if sparse_store is not None:
            sparse_residual = sparse_store.get(name, device=weight.device).to(weight.dtype)
            sparse_mask = sparse_residual != 0
            dense_weight = weight - sparse_residual
            total_sparse_values += int(sparse_mask.sum().item())

        cached_centers = codebook_store.get(name)
        codebook.set_context(CodebookContext(precomputed_centers=cached_centers))
        qres = codebook.quantize(dense_weight, row_chunk=row_chunk)
        gptq = NonUniformGPTQ(layer=module, qres=qres, reference_weight=dense_weight)

        def add_batch(_m, inp, out):
            x = inp[0] if isinstance(inp, tuple) else inp
            gptq.add_batch(x.data, out.data)

        handle = module.register_forward_hook(add_batch)
        try:
            for i, text in enumerate(calib_texts[:n_calib]):
                inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
                inputs = {k: v.to(device) for k, v in inputs.items()}
                model(**inputs, use_cache=False)
                if (i + 1) % 16 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
        finally:
            handle.remove()

        err, elapsed = gptq.fasterquant(
            blocksize=gptq_blocksize,
            percdamp=gptq_percdamp,
            actorder=gptq_act_order,
            freeze_mask=sparse_mask,
        )
        output = module.weight.data
        if sparse_residual is not None:
            output = output + sparse_residual
        module.weight.data = output.to(weight.dtype)
        codebook.set_context(None)
        gptq.free()

        total_error += err
        total_time_sec += elapsed
        print(f"GPTQ layer done | {name} | error={err:.6e} | time={elapsed:.2f}s")

    print(f"GPTQ summary | total_error={total_error:.6e} | total_time_sec={total_time_sec:.2f}")
    if sparse_store is not None:
        print(f"SqueezeLLM sparse summary | restored_values={total_sparse_values}")
    return {
        "method": "gptq",
        "num_linear_layers": len(linears),
        "skip_lmhead": skip_lmhead,
        "codebook": codebook.name,
        "bits": codebook.bits,
        "sparse_values": total_sparse_values,
        "total_error": total_error,
        "total_time_sec": total_time_sec,
        "gptq_blocksize": gptq_blocksize,
        "gptq_percdamp": gptq_percdamp,
        "gptq_act_order": gptq_act_order,
    }
