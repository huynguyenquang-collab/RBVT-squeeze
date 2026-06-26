"""Factory for codebook adapters backed by the nested upstream repositories."""

from __future__ import annotations

from .base_codebook import BaseCodebook


def get_codebook(
    name: str,
    bits: int,
    group_size: int = -1,
    leanquant_exponent: float = 4.0,
    leanquant_percdamp: float = 0.1,
    leanquant_act_order: bool = True,
    kmeans_seed: int = 0,
) -> BaseCodebook:
    key = name.lower().replace("-", "").replace("_", "")
    common = {
        "bits": bits,
        "group_size": group_size,
    }
    if key == "leanquant":
        return BaseCodebook(
            **common,
            name=f"leanquant{bits}",
            exponent=leanquant_exponent,
            percdamp=leanquant_percdamp,
            act_order=leanquant_act_order,
            kmeans_seed=kmeans_seed,
            gptq_block_size=128,
        )
    if key == "squeezellm":
        return BaseCodebook(
            **common,
            name=f"squeezellm{bits}",
        )
    raise ValueError("Unknown codebook {!r}. Available: leanquant, squeezellm".format(name))


__all__ = [
    "BaseCodebook",
    "get_codebook",
]
