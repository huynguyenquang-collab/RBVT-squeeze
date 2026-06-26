"""Quantizers and RBVT solver for RBVTQuant."""

from __future__ import annotations

from .base_quantizer import BaseQuantizer, QuantResult
from .learned_codebook import LearnedCodebookQuantizer
from .normalfloat import NormalFloatQuantizer
from .nvfp4 import NVFP4Quantizer
from .rbvt import RBVTStats, apply_rbvt


def _nf3(**kw):
    return NormalFloatQuantizer(bits=3, block_size=kw.get("nf_block_size", 64))


def _nf4(**kw):
    return NormalFloatQuantizer(bits=4, block_size=kw.get("nf_block_size", 64))


def _nvfp4(**kw):
    return NVFP4Quantizer(
        bits=4,
        block_size=kw.get("nvfp4_block_size", 16),
        fp8_scale=kw.get("fp8_scale", True),
    )


def _codebook3(**kw):
    return LearnedCodebookQuantizer(
        bits=3,
        block_size=kw.get("cb_block_size", 64),
        n_iters=kw.get("n_iters", 20),
        seed=kw.get("seed", 0),
    )


def _codebook4(**kw):
    return LearnedCodebookQuantizer(
        bits=4,
        block_size=kw.get("cb_block_size", 64),
        n_iters=kw.get("n_iters", 20),
        seed=kw.get("seed", 0),
    )


QUANTIZER_REGISTRY = {
    "nf3": _nf3,
    "nf4": _nf4,
    "nvfp4": _nvfp4,
    "codebook3": _codebook3,
    "codebook4": _codebook4,
}


def get_quantizer(name: str, **kwargs) -> BaseQuantizer:
    key = name.lower()
    if key not in QUANTIZER_REGISTRY:
        raise ValueError(f"Unknown quantizer {name!r}. Available: {sorted(QUANTIZER_REGISTRY)}")
    return QUANTIZER_REGISTRY[key](**kwargs)


__all__ = [
    "BaseQuantizer",
    "QuantResult",
    "NormalFloatQuantizer",
    "NVFP4Quantizer",
    "LearnedCodebookQuantizer",
    "RBVTStats",
    "apply_rbvt",
    "get_quantizer",
    "QUANTIZER_REGISTRY",
]
