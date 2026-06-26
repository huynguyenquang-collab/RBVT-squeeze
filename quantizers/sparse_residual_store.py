"""Disk-backed sparse residuals for SqueezeLLM dense-and-sparse quantization."""

from __future__ import annotations

import json
from pathlib import Path

import torch


class SparseResidualStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.manifest_path = self.root / "manifest.json"
        self._manifest = self._load_manifest()

    def _load_manifest(self) -> dict:
        if not self.manifest_path.exists():
            return {"complete": False, "layers": {}, "metadata": {}}
        return json.loads(self.manifest_path.read_text(encoding="utf-8"))

    def _write_manifest(self):
        self.manifest_path.write_text(
            json.dumps(self._manifest, indent=2, sort_keys=True),
            encoding="utf-8",
        )

    @property
    def complete(self) -> bool:
        return bool(self._manifest.get("complete", False))

    @property
    def metadata(self) -> dict:
        return dict(self._manifest.get("metadata", {}))

    def initialize(self, metadata: dict):
        if self._manifest.get("layers") and self.metadata != metadata:
            raise ValueError(
                f"Sparse residual cache metadata mismatch at {self.root}: "
                f"found {self.metadata}, requested {metadata}"
            )
        self._manifest["metadata"] = metadata
        self._manifest["complete"] = False
        self._write_manifest()

    def validate(self, metadata: dict):
        if self.metadata != metadata:
            raise ValueError(
                f"Sparse residual cache metadata mismatch at {self.root}: "
                f"found {self.metadata}, requested {metadata}"
            )

    @staticmethod
    def _filename(layer_name: str) -> str:
        return layer_name.replace("/", "__").replace(".", "_") + ".pt"

    def put(self, layer_name: str, residual: torch.Tensor):
        filename = self._filename(layer_name)
        torch.save(residual.detach().float().cpu().to_sparse(), self.root / filename)
        self._manifest.setdefault("layers", {})[layer_name] = filename
        self._write_manifest()

    def get(self, layer_name: str, device=None) -> torch.Tensor:
        filename = self._manifest.get("layers", {}).get(layer_name)
        if filename is None:
            raise KeyError(f"No sparse residual cached for {layer_name!r}")
        try:
            residual = torch.load(
                self.root / filename,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            residual = torch.load(self.root / filename, map_location="cpu")
        residual = residual.to_dense() if residual.is_sparse else residual
        return residual.to(device) if device is not None else residual

    def mark_complete(self):
        self._manifest["complete"] = True
        self._write_manifest()


__all__ = ["SparseResidualStore"]
