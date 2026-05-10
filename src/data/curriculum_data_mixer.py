#!/usr/bin/env python3
"""Multi-stage curriculum data mixer.

Reads ``configs/curriculum.yaml`` and produces weighted data mixtures
that shift as training progresses through stages.  Integrates with
``TokenizedShardDataset`` by returning stage-aware shard paths and
sampling weights.

Usage:
    from src.data.curriculum_data_mixer import CurriculumMixer
    mixer = CurriculumMixer("configs/curriculum.yaml")
    stage = mixer.get_stage_for_tokens(tokens_seen)
    shard_paths, weights = mixer.get_mix_for_stage(stage["name"])
"""

from __future__ import annotations

import logging
import random
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)


class CurriculumMixer:
    """Curriculum data mixer driven by a YAML stage specification."""

    def __init__(self, config_path: str | Path, base_data_dir: str | Path = "data/pretrain") -> None:
        self.config_path = Path(config_path)
        self.base_data_dir = Path(base_data_dir)
        self.stages: list[dict[str, Any]] = []
        self._load_config()

    def _load_config(self) -> None:
        with open(self.config_path) as f:
            data = yaml.safe_load(f)
        self.stages = data.get("stages", [])
        logger.info("Loaded %d curriculum stages from %s", len(self.stages), self.config_path)

    def get_stage_for_tokens(self, tokens_seen: int) -> dict[str, Any]:
        """Return the stage dict that covers the given token count."""
        for stage in self.stages:
            low, high = stage["token_range"]
            if low <= tokens_seen < high:
                return stage
        # Default to last stage if past all ranges
        return self.stages[-1] if self.stages else {}

    def get_mix_for_stage(self, stage_name: str) -> tuple[list[Path], list[float]]:
        """Return (shard_paths, weights) for a named stage.

        Shard paths are discovered under ``base_data_dir/<domain>/``.
        Weights are normalized to sum to 1.0.
        """
        stage = None
        for s in self.stages:
            if s["name"] == stage_name:
                stage = s
                break
        if stage is None:
            raise ValueError(f"Stage '{stage_name}' not found in curriculum config")

        data_mix: dict[str, float] = stage.get("data_mix", {})
        shard_paths: list[Path] = []
        weights: list[float] = []

        for domain, weight in data_mix.items():
            domain_dir = self.base_data_dir / domain
            if not domain_dir.exists():
                logger.warning("Domain directory not found: %s", domain_dir)
                continue
            shards = sorted(domain_dir.glob("*.npy"))
            if not shards:
                logger.warning("No .npy shards in %s", domain_dir)
                continue
            shard_paths.extend(shards)
            # Each shard in this domain gets the domain's weight
            weights.extend([weight] * len(shards))

        if not shard_paths:
            raise FileNotFoundError(f"No shards found for stage '{stage_name}'")

        # Normalize weights
        total = sum(weights)
        weights = [w / total for w in weights]
        return shard_paths, weights

    def sample_shards_for_step(
        self,
        stage_name: str,
        n_shards: int,
        seed: int | None = None,
    ) -> list[Path]:
        """Sample ``n_shards`` paths according to stage mixture weights."""
        shard_paths, weights = self.get_mix_for_stage(stage_name)
        if seed is not None:
            random.seed(seed)
        sampled = random.choices(shard_paths, weights=weights, k=n_shards)
        return sampled

    def get_lr_for_stage(self, stage_name: str) -> dict[str, Any]:
        """Return LR schedule parameters for the named stage."""
        for stage in self.stages:
            if stage["name"] == stage_name:
                return stage.get("lr_schedule", {})
        return {}


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Inspect curriculum mixer state")
    parser.add_argument("--config", default="configs/curriculum.yaml")
    parser.add_argument("--tokens", type=int, default=0, help="Simulated tokens seen")
    parser.add_argument("--base-dir", default="data/pretrain")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    mixer = CurriculumMixer(args.config, args.base_dir)

    stage = mixer.get_stage_for_tokens(args.tokens)
    print(f"Tokens seen: {args.tokens:,}")
    print(f"Active stage: {stage.get('name', 'unknown')}")
    print(f"Token range: {stage.get('token_range', [])}")
    print(f"Data mix: {stage.get('data_mix', {})}")
    print(f"LR schedule: {stage.get('lr_schedule', {})}")

    try:
        shards, weights = mixer.get_mix_for_stage(stage["name"])
        print(f"\nShards: {len(shards)}")
        print(f"Weights (first 10): {weights[:10]}")
    except FileNotFoundError as e:
        print(f"\nNo shards available yet: {e}")


if __name__ == "__main__":
    main()
