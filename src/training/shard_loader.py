from __future__ import annotations

import json
from pathlib import Path

import torch

from src.training.checkpoint_sharder import CheckpointSharder, ShardManifest

_DTYPE_MAP: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "int32": torch.int32,
    "int64": torch.int64,
    "int8": torch.int8,
    "uint8": torch.uint8,
    "bool": torch.bool,
}


class ShardLoader:
    """Load sharded checkpoint back into a model, shard by shard."""

    def __init__(self, sharder: CheckpointSharder | None = None) -> None:
        self._sharder = sharder or CheckpointSharder()

    def load_manifest(self, checkpoint_dir: str) -> ShardManifest:
        """Read manifest.json from directory."""
        return self._sharder.manifest_from_dir(checkpoint_dir)

    def load_shard(
        self,
        checkpoint_dir: str,
        shard_index: int,
        manifest: ShardManifest,
    ) -> dict[str, torch.Tensor]:
        """Load one shard; reconstructs tensors as zeros for JSON stub files."""
        filename = manifest.shard_files[shard_index]
        path = Path(checkpoint_dir) / filename
        stub = json.loads(path.read_text())
        result: dict[str, torch.Tensor] = {}
        for key, meta in stub.items():
            dtype = _DTYPE_MAP.get(meta["dtype"], torch.float32)
            result[key] = torch.zeros(meta["shape"], dtype=dtype)
        return result

    def load_state_dict(self, checkpoint_dir: str) -> dict[str, torch.Tensor]:
        """Load all shards, merge into single state_dict."""
        manifest = self.load_manifest(checkpoint_dir)
        merged: dict[str, torch.Tensor] = {}
        for i in range(manifest.n_shards):
            merged.update(self.load_shard(checkpoint_dir, i, manifest))
        return merged

    def load_into_model(
        self,
        model: torch.nn.Module,
        checkpoint_dir: str,
        strict: bool = True,
    ) -> tuple[list[str], list[str]]:
        """Load merged state_dict into model; returns (missing_keys, unexpected_keys)."""
        state = self.load_state_dict(checkpoint_dir)
        result = model.load_state_dict(state, strict=strict)
        return list(result.missing_keys), list(result.unexpected_keys)

    def partial_load(
        self,
        model: torch.nn.Module,
        checkpoint_dir: str,
        key_pattern: str,
    ) -> int:
        """Load only keys whose name contains key_pattern; return count loaded."""
        state = self.load_state_dict(checkpoint_dir)
        filtered = {k: v for k, v in state.items() if key_pattern in k}
        model_state = model.state_dict()
        model_state.update(filtered)
        model.load_state_dict(model_state, strict=True)
        return len(filtered)
