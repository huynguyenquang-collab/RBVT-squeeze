"""
Relaxed Bias-Variance Transport (RBVT) assignment solver.

This keeps NCCQuant's block/codebook representation, but replaces the final
greedy correction by the soft relaxation in `RBVT_soft_relaxation_note.md`:

1. start from nearest-codeword assignment;
2. build sign-aligned neighbouring moves;
3. solve the sorted diagonal fractional relaxation per output channel;
4. project the relaxed solution back to discrete neighbour assignments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from .base_quantizer import QuantResult


@dataclass
class RBVTStats:
    flips: int
    channels: int
    candidates: int
    boundary_kept: int
    bias_before: float
    bias_after: float
    objective_before: float
    objective_after: float
    variance_increase: float


@torch.no_grad()
def _block_realised_levels(qres: QuantResult, r0: int, r1: int, device: torch.device) -> torch.Tensor:
    if qres.block_codebooks is not None:
        return qres.block_codebooks[r0:r1].to(device).float()
    q = qres.q_levels.to(device).float()
    bscale = qres.block_scales[r0:r1].to(device).float()
    return bscale.unsqueeze(-1) * q.view(1, 1, -1)


@torch.no_grad()
def apply_rbvt(
    W_fp: torch.Tensor,
    qres: QuantResult,
    mu: torch.Tensor,
    sigma_ii: Optional[torch.Tensor] = None,
    rbvt_lambda: float = 1.0,
    rbvt_topk: Optional[int] = None,
    row_chunk: int = 1024,
    gap_floor: float = 1e-8,
    relax_eps: float = 1e-12,
    strict_descent: bool = True,
    candidate_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, RBVTStats]:
    if rbvt_lambda < 0.0:
        raise ValueError(f"rbvt_lambda must be non-negative, got {rbvt_lambda}")

    device = W_fp.device
    out_features, in_features = W_fp.shape
    bs = qres.block_size

    mu = mu.to(device).float()
    if sigma_ii is None:
        sigma_ii = torch.zeros_like(mu)
    else:
        sigma_ii = sigma_ii.to(device).float()

    Wq_full = qres.W_dequant.to(device).float()
    indices_full = qres.indices.to(device)
    Wq_rbvt = Wq_full.clone()
    if candidate_mask is not None:
        candidate_mask = candidate_mask.to(device=device, dtype=torch.bool)
        if candidate_mask.shape != W_fp.shape:
            raise ValueError(
                f"candidate_mask has shape {tuple(candidate_mask.shape)}, "
                f"expected {tuple(W_fp.shape)}"
            )

    total_flips = 0
    total_candidates = 0
    boundary_kept = 0
    bias_before = 0.0
    bias_after = 0.0
    objective_before = 0.0
    objective_after = 0.0
    variance_increase = 0.0

    col_block = torch.arange(in_features, device=device) // bs

    for r0 in range(0, out_features, row_chunk):
        r1 = min(r0 + row_chunk, out_features)
        rc = r1 - r0

        Wr = W_fp[r0:r1].float()
        Wq = Wq_full[r0:r1]
        idx = indices_full[r0:r1]
        levels = _block_realised_levels(qres, r0, r1, device)
        L = levels.shape[-1]

        e = Wq - Wr
        e_sign = torch.sign(e)
        b = e @ mu

        left_idx = (idx - 1).clamp(min=0)
        right_idx = (idx + 1).clamp(max=L - 1)
        row_ids = torch.arange(rc, device=device).unsqueeze(1)
        blk_ids = col_block.unsqueeze(0).expand(rc, in_features)

        # Read only the realised current / neighbouring levels that RBVT needs,
        # instead of materialising the full [rows, in_features, L] table.
        cur = levels[row_ids, blk_ids, idx]
        left = levels[row_ids, blk_ids, left_idx]
        right = levels[row_ids, blk_ids, right_idx]

        g_left = (cur - left).abs()
        g_right = (right - cur).abs()
        move_down = e_sign > 0
        gap = torch.where(move_down, g_left, g_right)
        target_val = torch.where(move_down, left, right)
        feasible = torch.where(move_down, idx > 0, idx < (L - 1))
        gap_ok = gap > gap_floor

        v = mu.unsqueeze(0) * e_sign * gap
        r = v.abs()
        q = sigma_ii.unsqueeze(0) * (gap.square() - 2.0 * gap * e.abs()).clamp(min=0.0)

        sign_aligned = (b.unsqueeze(1) * v) > 0
        admissible = feasible & gap_ok & sign_aligned & (r > relax_eps)
        if candidate_mask is not None:
            admissible &= candidate_mask[r0:r1]
        rho = q / (r + relax_eps)

        for rr in range(rc):
            T = float(abs(b[rr].item()))
            base_obj = T * T
            bias_before += base_obj
            objective_before += base_obj

            if T <= relax_eps:
                bias_after += base_obj
                objective_after += base_obj
                continue

            cand = torch.nonzero(admissible[rr], as_tuple=False).squeeze(1)
            total_candidates += int(cand.numel())
            if cand.numel() == 0:
                bias_after += base_obj
                objective_after += base_obj
                continue

            cand_rho = rho[rr, cand]
            if rbvt_topk is not None and rbvt_topk > 0 and cand.numel() > rbvt_topk:
                _, topk_idx = torch.topk(cand_rho, k=rbvt_topk, largest=False, sorted=False)
                cand = cand[topk_idx]
                cand_rho = cand_rho[topk_idx]

            cand_order = torch.argsort(cand_rho, descending=False)
            cand = cand[cand_order]

            r_cand = r[rr, cand]
            q_cand = q[rr, cand]
            limit = T if strict_descent else 2.0 * T

            cum_r = torch.cumsum(r_cand, dim=0)
            cum_q = torch.cumsum(q_cand, dim=0)
            zero = torch.zeros(1, device=device, dtype=r_cand.dtype)
            s_prev = torch.cat([zero, cum_r[:-1]], dim=0)
            q_prev = torch.cat([zero, cum_q[:-1]], dim=0)

            upper = ((limit - s_prev) / (r_cand + relax_eps)).clamp(min=0.0, max=1.0)
            gamma_star = (T - s_prev - rbvt_lambda * q_cand / (2.0 * (r_cand + relax_eps))) / (r_cand + relax_eps)
            gamma = torch.minimum(torch.maximum(gamma_star, torch.zeros_like(gamma_star)), upper)

            relaxed_obj = (T - s_prev - gamma * r_cand).square() + rbvt_lambda * (q_prev + gamma * q_cand)
            relaxed_obj = torch.where(upper > 0.0, relaxed_obj, torch.full_like(relaxed_obj, float("inf")))

            best_val, best_pos = relaxed_obj.min(dim=0)
            if float(best_val.item()) >= base_obj:
                bias_after += base_obj
                objective_after += base_obj
                continue

            best_pos_i = int(best_pos.item())
            best_gamma = float(gamma[best_pos_i].item())
            prefix_count = best_pos_i
            prefix_r = float(s_prev[best_pos_i].item())
            prefix_q = float(q_prev[best_pos_i].item())

            drop_obj = (T - prefix_r) ** 2 + rbvt_lambda * prefix_q
            keep_valid = best_gamma > 0.0
            keep_obj = float("inf")
            keep_count = prefix_count
            keep_r = prefix_r
            keep_q = prefix_q
            if keep_valid:
                keep_r = float((prefix_r + r_cand[best_pos_i]).item())
                if keep_r <= limit + 1e-8:
                    keep_q = float((prefix_q + q_cand[best_pos_i]).item())
                    keep_obj = (T - keep_r) ** 2 + rbvt_lambda * keep_q
                    keep_count = prefix_count + 1

            if keep_obj < drop_obj:
                chosen_count = keep_count
                boundary_kept += int(best_gamma < 1.0)
                final_r = keep_r
                final_q = keep_q
            else:
                chosen_count = prefix_count
                final_r = prefix_r
                final_q = prefix_q

            if chosen_count > 0:
                chosen = cand[:chosen_count]
                Wq_rbvt[r0 + rr, chosen] = target_val[rr, chosen]
                total_flips += chosen_count

            bias_after += (T - final_r) ** 2
            objective_after += (T - final_r) ** 2 + rbvt_lambda * final_q
            variance_increase += final_q

    return Wq_rbvt, RBVTStats(
        flips=total_flips,
        channels=out_features,
        candidates=total_candidates,
        boundary_kept=boundary_kept,
        bias_before=bias_before,
        bias_after=bias_after,
        objective_before=objective_before,
        objective_after=objective_after,
        variance_increase=variance_increase,
    )
