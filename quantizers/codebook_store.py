"""Disk-backed per-layer codebook cache shared by RTN and RBVT runs."""

from __future__ import annotations

import json
from pathlib import Path

import torch


class CodebookStore:
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
                f"Codebook cache metadata mismatch at {self.root}: "
                f"found {self.metadata}, requested {metadata}"
            )
        self._manifest["metadata"] = metadata
        self._manifest["complete"] = False
        self._write_manifest()

    def validate(self, metadata: dict):
        if self.metadata != metadata:
            raise ValueError(
                f"Codebook cache metadata mismatch at {self.root}: "
                f"found {self.metadata}, requested {metadata}"
            )

    @staticmethod
    def _filename(layer_name: str) -> str:
        return layer_name.replace("/", "__").replace(".", "_") + ".pt"

    def has(self, layer_name: str) -> bool:
        filename = self._manifest.get("layers", {}).get(layer_name)
        return filename is not None and (self.root / filename).exists()

    def put(self, layer_name: str, centers: torch.Tensor):
        filename = self._filename(layer_name)
        torch.save(centers.detach().float().cpu(), self.root / filename)
        self._manifest.setdefault("layers", {})[layer_name] = filename
        self._write_manifest()

    def get(self, layer_name: str) -> torch.Tensor | None:
        filename = self._manifest.get("layers", {}).get(layer_name)
        if filename is None:
            return None
        path = self.root / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing cached codebook for {layer_name!r}: {path}")
        try:
            return torch.load(path, map_location="cpu", weights_only=True)
        except TypeError:
            return torch.load(path, map_location="cpu")

    def mark_complete(self):
        self._manifest["complete"] = True
        self._write_manifest()


__all__ = ["CodebookStore"]
