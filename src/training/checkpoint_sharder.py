from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import torch


@dataclass
class ShardConfig:
    max_shard_size_gb: float = 5.0
    format: str = "safetensors"
    prefix: str = "model"


@dataclass
class ShardManifest:
    total_params: int
    total_size_bytes: int
    n_shards: int
    shard_files: list[str]
    key_to_shard: dict[str, int]
    format: str


class CheckpointSharder:
    """Split model state_dict into ≤max_shard_size shards."""

    def __init__(self, config: ShardConfig | None = None) -> None:
        self.config = config or ShardConfig()

    def _estimate_size(self, tensor: torch.Tensor) -> int:
        return tensor.nelement() * tensor.element_size()

    def split_plan(self, state_dict: dict[str, torch.Tensor]) -> list[list[str]]:
        """Return list of key groups per shard (dry run, no file I/O)."""
        max_bytes = int(self.config.max_shard_size_gb * 1024**3)
        sorted_keys = sorted(state_dict.keys())

        shards: list[list[str]] = [[]]
        current_size = 0

        for key in sorted_keys:
            size = self._estimate_size(state_dict[key])
            if current_size + size > max_bytes and shards[-1]:
                shards.append([])
                current_size = 0
            shards[-1].append(key)
            current_size += size

        return shards

    def shard(
        self,
        state_dict: dict[str, torch.Tensor],
        output_dir: str,
    ) -> ShardManifest:
        """Write shards and manifest to output_dir; shards stored as JSON stubs."""
        os.makedirs(output_dir, exist_ok=True)
        plan = self.split_plan(state_dict)
        n = len(plan)
        shard_files: list[str] = []
        key_to_shard: dict[str, int] = {}
        total_params = 0
        total_bytes = 0

        for i, keys in enumerate(plan):
            filename = f"{self.config.prefix}-{i:05d}-of-{n:05d}.safetensors"
            shard_files.append(filename)
            stub: dict[str, object] = {}
            for key in keys:
                t = state_dict[key]
                stub[key] = {
                    "shape": list(t.shape),
                    "dtype": str(t.dtype).replace("torch.", ""),
                }
                key_to_shard[key] = i
                total_params += t.nelement()
                total_bytes += self._estimate_size(t)
            shard_path = Path(output_dir) / filename
            shard_path.write_text(json.dumps(stub))

        manifest = ShardManifest(
            total_params=total_params,
            total_size_bytes=total_bytes,
            n_shards=n,
            shard_files=shard_files,
            key_to_shard=key_to_shard,
            format=self.config.format,
        )
        manifest_path = Path(output_dir) / "manifest.json"
        manifest_path.write_text(
            json.dumps(
                {
                    "total_params": manifest.total_params,
                    "total_size_bytes": manifest.total_size_bytes,
                    "n_shards": manifest.n_shards,
                    "shard_files": manifest.shard_files,
                    "key_to_shard": manifest.key_to_shard,
                    "format": manifest.format,
                }
            )
        )
        return manifest

    def manifest_from_dir(self, output_dir: str) -> ShardManifest:
        """Read manifest.json from directory."""
        data = json.loads((Path(output_dir) / "manifest.json").read_text())
        return ShardManifest(
            total_params=data["total_params"],
            total_size_bytes=data["total_size_bytes"],
            n_shards=data["n_shards"],
            shard_files=data["shard_files"],
            key_to_shard=data["key_to_shard"],
            format=data["format"],
        )
