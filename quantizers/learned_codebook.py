"""Learned scalar codebook quantizer reused from NCCQuant."""

from __future__ import annotations

import torch

from .base_quantizer import BaseQuantizer, QuantResult


class LearnedCodebookQuantizer(BaseQuantizer):
    def __init__(self, bits: int, block_size: int = 64, n_iters: int = 20, seed: int = 0):
        if bits not in (3, 4):
            raise ValueError(f"LearnedCodebook supports bits in {{3,4}}, got {bits}")
        super().__init__(bits=bits, block_size=block_size)
        self.name = f"codebook{bits}"
        self.num_levels = 2 ** bits
        self.n_iters = n_iters
        self.seed = seed

    @property
    def q_levels(self) -> torch.Tensor:
        return torch.linspace(-1.0, 1.0, self.num_levels)

    @torch.no_grad()
    def _kmeans_blocks(self, Wb: torch.Tensor) -> torch.Tensor:
        G, _ = Wb.shape
        K = self.num_levels
        device = Wb.device

        qs = torch.linspace(0.0, 1.0, K, device=device, dtype=torch.float32)
        centers = torch.quantile(Wb, qs, dim=1).t().contiguous()
        zc = centers.abs().argmin(dim=1)
        centers[torch.arange(G, device=device), zc] = 0.0
        centers, _ = torch.sort(centers, dim=1)

        for _ in range(self.n_iters):
            d = (Wb.unsqueeze(-1) - centers.unsqueeze(1)).abs()
            assign = d.argmin(dim=-1)
            del d
            new_centers = centers.clone()
            for k in range(K):
                mask = assign == k
                cnt = mask.sum(dim=1).clamp(min=1)
                summ = (Wb * mask).sum(dim=1)
                mean_k = summ / cnt
                owns = mask.any(dim=1)
                new_centers[:, k] = torch.where(owns, mean_k, centers[:, k])
            centers, _ = torch.sort(new_centers, dim=1)

        eps = 1e-7
        for k in range(1, K):
            bad = centers[:, k] <= centers[:, k - 1]
            centers[:, k] = torch.where(bad, centers[:, k - 1] + eps, centers[:, k])
        return centers

    @torch.no_grad()
    def quantize(self, W: torch.Tensor, row_chunk: int = 1024) -> QuantResult:
        torch.manual_seed(self.seed)
        device = W.device
        out_features, in_features = W.shape
        K = self.num_levels
        bs = self.block_size
        n_blocks = (in_features + bs - 1) // bs

        W_dequant = torch.empty_like(W)
        indices = torch.empty(out_features, in_features, dtype=torch.long, device=device)
        block_codebooks = torch.zeros(out_features, n_blocks, K, device=device, dtype=torch.float32)
        block_scales = torch.zeros(out_features, n_blocks, device=device, dtype=torch.float32)

        for r0 in range(0, out_features, row_chunk):
            r1 = min(r0 + row_chunk, out_features)
            Wr = W[r0:r1].float()
            for b in range(n_blocks):
                c0 = b * bs
                c1 = min(c0 + bs, in_features)
                Wb = Wr[:, c0:c1]
                centers = self._kmeans_blocks(Wb)
                block_codebooks[r0:r1, b, :] = centers
                block_scales[r0:r1, b] = Wb.abs().amax(dim=1)

                diff = (Wb.unsqueeze(-1) - centers.unsqueeze(1)).abs()
                idx = diff.argmin(dim=-1)
                deq = torch.gather(centers, 1, idx)
                W_dequant[r0:r1, c0:c1] = deq.to(W.dtype)
                indices[r0:r1, c0:c1] = idx
                del diff

        return QuantResult(
            W_dequant=W_dequant,
            indices=indices,
            q_levels=self.q_levels.to(device),
            block_scales=block_scales,
            block_size=bs,
            block_codebooks=block_codebooks,
            block_zeros=None,
        )
