"""Lazy access to SqueezeLLM Fisher sensitivity checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


class SensitivityStore:
    """Load one layer sensitivity tensor at a time from common checkpoints."""

    def __init__(self, path: str | Path | None):
        self.path = Path(path).expanduser() if path else None
        self._weight_map: dict[str, str] = {}
        self._tensor_map: dict[str, str] = {}
        self._single_safetensors: Path | None = None
        self._torch_payload: Any = None

        if self.path is None:
            return
        if not self.path.exists():
            raise FileNotFoundError(f"Sensitivity checkpoint not found: {self.path}")

        if self.path.is_dir():
            manifest_path = self.path / "manifest.json"
            if manifest_path.exists():
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                if not payload.get("complete", False):
                    raise ValueError(f"Incomplete Fisher cache: {self.path}")
                self._tensor_map = dict(payload.get("layers", {}))
                return
            indexes = sorted(self.path.glob("*.safetensors.index.json"))
            if indexes:
                payload = json.loads(indexes[0].read_text(encoding="utf-8"))
                self._weight_map = dict(payload.get("weight_map", {}))
                return

            safetensors_files = sorted(self.path.glob("*.safetensors"))
            if len(safetensors_files) == 1:
                self._single_safetensors = safetensors_files[0]
                return
            raise ValueError(
                "Sensitivity directory must contain a safetensors index or one "
                f".safetensors file: {self.path}"
            )

        if self.path.suffix == ".safetensors":
            self._single_safetensors = self.path
            return

        try:
            self._torch_payload = torch.load(
                self.path,
                map_location="cpu",
                weights_only=True,
            )
        except TypeError:
            self._torch_payload = torch.load(self.path, map_location="cpu")

    @property
    def mode(self) -> str:
        return "fisher_checkpoint" if self.path is not None else "missing"

    @staticmethod
    def _candidate_keys(layer_name: str) -> tuple[str, ...]:
        return (
            f"{layer_name}.weight",
            layer_name,
            f"model.{layer_name}.weight",
            f"model.{layer_name}",
        )

    def _load_safetensor(self, file_path: Path, key: str) -> torch.Tensor:
        try:
            from safetensors import safe_open
        except ImportError as exc:
            raise RuntimeError(
                "safetensors is required for a SqueezeLLM sensitivity checkpoint"
            ) from exc

        with safe_open(file_path, framework="pt", device="cpu") as handle:
            return handle.get_tensor(key)

    def get(
        self,
        layer_name: str,
        expected_shape: torch.Size | tuple[int, ...],
    ) -> torch.Tensor | None:
        if self.path is None:
            return None

        candidates = self._candidate_keys(layer_name)
        tensor: torch.Tensor | None = None

        if self._tensor_map:
            for key in candidates:
                filename = self._tensor_map.get(key)
                if filename is not None:
                    file_path = self.path / filename
                    try:
                        tensor = torch.load(
                            file_path,
                            map_location="cpu",
                            weights_only=True,
                        )
                    except TypeError:
                        tensor = torch.load(file_path, map_location="cpu")
                    break
        elif self._weight_map:
            for key in candidates:
                shard = self._weight_map.get(key)
                if shard is not None:
                    tensor = self._load_safetensor(self.path / shard, key)
                    break
        elif self._single_safetensors is not None:
            try:
                from safetensors import safe_open
            except ImportError as exc:
                raise RuntimeError(
                    "safetensors is required for a SqueezeLLM sensitivity checkpoint"
                ) from exc
            with safe_open(self._single_safetensors, framework="pt", device="cpu") as handle:
                available = set(handle.keys())
                for key in candidates:
                    if key in available:
                        tensor = handle.get_tensor(key)
                        break
        else:
            payload = self._torch_payload
            if isinstance(payload, dict) and "state_dict" in payload:
                payload = payload["state_dict"]
            if isinstance(payload, dict):
                for key in candidates:
                    value = payload.get(key)
                    if isinstance(value, torch.Tensor):
                        tensor = value
                        break

        if tensor is None:
            raise KeyError(
                f"No sensitivity tensor found for layer {layer_name!r} in {self.path}"
            )
        if tuple(tensor.shape) != tuple(expected_shape):
            raise ValueError(
                f"Sensitivity for {layer_name!r} has shape {tuple(tensor.shape)}, "
                f"expected {tuple(expected_shape)}"
            )
        return tensor.float()
