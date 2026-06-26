"""NF3/NF4 block-wise quantizers reused from NCCQuant."""

from __future__ import annotations

import torch

from .base_quantizer import BaseQuantizer


_NF4_LEVELS = [
    -1.0,
    -0.6961928009986877,
    -0.5250730514526367,
    -0.39491748809814453,
    -0.28444138169288635,
    -0.18477343022823334,
    -0.09105003625154495,
    0.0,
    0.07958029955625534,
    0.16093020141124725,
    0.24611230194568634,
    0.33791524171829224,
    0.44070982933044434,
    0.5626170039176941,
    0.7229568362236023,
    1.0,
]


def _make_normalfloat_levels(num_levels: int) -> torch.Tensor:
    normal = torch.distributions.Normal(0.0, 1.0)
    ps = (torch.arange(num_levels, dtype=torch.float64) + 0.5) / num_levels
    levels = normal.icdf(ps)
    zero_idx = int(levels.abs().argmin())
    levels[zero_idx] = 0.0
    levels = levels / levels.abs().max().clamp(min=1e-12)
    levels, _ = torch.sort(levels)
    return levels.to(torch.float32)


class NormalFloatQuantizer(BaseQuantizer):
    def __init__(self, bits: int, block_size: int = 64):
        if bits not in (3, 4):
            raise ValueError(f"NormalFloat supports bits in {{3,4}}, got {bits}")
        super().__init__(bits=bits, block_size=block_size)
        self.name = f"nf{bits}"
        q = torch.tensor(_NF4_LEVELS, dtype=torch.float32) if bits == 4 else _make_normalfloat_levels(2 ** bits)
        q, _ = torch.sort(q)
        self._q = q / q.abs().max().clamp(min=1e-12)

    @property
    def q_levels(self) -> torch.Tensor:
        return self._q
