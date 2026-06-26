"""
Base interface for non-uniform per-block scalar codebook quantizers.

Copied from NCCQuant so RBVTQuant keeps the same block/codebook backbone while
changing only the assignment optimisation stage.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch


ASYM: bool = True


@dataclass
class QuantResult:
    W_dequant: torch.Tensor
    indices: torch.Tensor
    q_levels: torch.Tensor
    block_scales: torch.Tensor
    block_size: int
    block_codebooks: Optional[torch.Tensor] = None
    block_zeros: Optional[torch.Tensor] = None


class BaseQuantizer(ABC):
    name: str = "base"

    def __init__(self, bits: int, block_size: int = 64):
        self.bits = bits
        self.block_size = block_size

    @property
    @abstractmethod
    def q_levels(self) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def quantize(self, W: torch.Tensor, row_chunk: int = 1024) -> QuantResult:
        device = W.device
        out_features, in_features = W.shape
        q = self.q_levels.to(device).float()
        L = q.numel()
        qmax = q.abs().max().clamp(min=1e-12)
        qlo = q.min()
        qhi = q.max()
        qspan = (qhi - qlo).clamp(min=1e-12)
        bs = self.block_size
        n_blocks = (in_features + bs - 1) // bs

        W_dequant = torch.empty_like(W)
        indices = torch.empty(out_features, in_features, dtype=torch.long, device=device)
        block_scales = torch.empty(out_features, n_blocks, device=device, dtype=torch.float32)
        block_zeros = (
            torch.zeros(out_features, n_blocks, device=device, dtype=torch.float32)
            if ASYM else None
        )
        block_codebooks = (
            torch.zeros(out_features, n_blocks, L, device=device, dtype=torch.float32)
            if ASYM else None
        )

        for r0 in range(0, out_features, row_chunk):
            r1 = min(r0 + row_chunk, out_features)
            Wr = W[r0:r1].float()
            rc = r1 - r0
            for b in range(n_blocks):
                c0 = b * bs
                c1 = min(c0 + bs, in_features)
                Wb = Wr[:, c0:c1]

                if ASYM:
                    wmin = Wb.amin(dim=1, keepdim=True)
                    wmax = Wb.amax(dim=1, keepdim=True)
                    scale = ((wmax - wmin) / qspan).clamp(min=1e-12)
                    z = qlo - wmin / scale
                    block_scales[r0:r1, b] = scale.squeeze(1)
                    block_zeros[r0:r1, b] = z.squeeze(1)
                    grid = scale * (q.unsqueeze(0) - z)
                    block_codebooks[r0:r1, b, :] = grid
                else:
                    absmax = Wb.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
                    scale = absmax / qmax
                    block_scales[r0:r1, b] = scale.squeeze(1)
                    grid = scale * q.unsqueeze(0)

                diff = (Wb.unsqueeze(-1) - grid.unsqueeze(1)).abs()
                idx = diff.argmin(dim=-1)
                deq = torch.gather(grid, 1, idx)
                W_dequant[r0:r1, c0:c1] = deq.to(W.dtype)
                indices[r0:r1, c0:c1] = idx
                del diff, grid

        return QuantResult(
            W_dequant=W_dequant,
            indices=indices,
            q_levels=q,
            block_scales=block_scales,
            block_size=bs,
            block_codebooks=block_codebooks,
            block_zeros=block_zeros,
        )

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(name={self.name!r}, bits={self.bits}, "
            f"block_size={self.block_size}, asym={ASYM})"
        )
