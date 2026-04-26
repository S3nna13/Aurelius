"""Red-team dataset generator for safety evaluation of language models.

Produces synthetic adversarial prompts via token-level perturbations, prefix/suffix
injection patterns, and random token noise to build a diverse evaluation set.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
from torch import LongTensor, Tensor


@dataclass
class RedTeamSample:
    """Single red-team evaluation sample."""

    prompt_ids: LongTensor
    label: str
    perturbation_type: str


@dataclass
class RedTeamConfig:
    """Configuration for the red-team dataset generator."""

    n_benign: int = 10
    n_adversarial: int = 10
    max_len: int = 32
    vocab_size: int = 256
    seed: int = 42


_PREFIX_PATTERN: list[int] = [1, 2, 3, 4]
_SUFFIX_PATTERN: list[int] = [5, 6, 7, 8]
_PATTERN_LEN: int = 4


class RedTeamGenerator:
    """Generates synthetic adversarial prompt datasets for safety evaluation."""

    def __init__(self, config: RedTeamConfig) -> None:
        self.config = config
        torch.manual_seed(config.seed)
        random.seed(config.seed)
        self._rng = torch.Generator()
        self._rng.manual_seed(config.seed)

    def _random_ids(self, length: int) -> LongTensor:
        ids = torch.randint(
            low=0,
            high=self.config.vocab_size,
            size=(1, length),
            generator=self._rng,
            dtype=torch.long,
        )
        return ids

    def generate_benign(self, n: int) -> list[RedTeamSample]:
        samples: list[RedTeamSample] = []
        for _ in range(n):
            ids = self._random_ids(self.config.max_len)
            samples.append(
                RedTeamSample(
                    prompt_ids=ids,
                    label="benign",
                    perturbation_type="none",
                )
            )
        return samples

    def generate_prefix_injection(self, n: int) -> list[RedTeamSample]:
        samples: list[RedTeamSample] = []
        tail_len = self.config.max_len - _PATTERN_LEN
        prefix = torch.tensor(_PREFIX_PATTERN, dtype=torch.long).unsqueeze(0)
        for _ in range(n):
            tail = self._random_ids(tail_len)
            ids = torch.cat([prefix, tail], dim=1)
            samples.append(
                RedTeamSample(
                    prompt_ids=ids,
                    label="adversarial",
                    perturbation_type="prefix_injection",
                )
            )
        return samples

    def generate_suffix_injection(self, n: int) -> list[RedTeamSample]:
        samples: list[RedTeamSample] = []
        head_len = self.config.max_len - _PATTERN_LEN
        suffix = torch.tensor(_SUFFIX_PATTERN, dtype=torch.long).unsqueeze(0)
        for _ in range(n):
            head = self._random_ids(head_len)
            ids = torch.cat([head, suffix], dim=1)
            samples.append(
                RedTeamSample(
                    prompt_ids=ids,
                    label="adversarial",
                    perturbation_type="suffix_injection",
                )
            )
        return samples

    def generate_random_token_attack(
        self, n: int, noise_fraction: float = 0.3
    ) -> list[RedTeamSample]:
        samples: list[RedTeamSample] = []
        n_noise = max(1, int(self.config.max_len * noise_fraction))
        for _ in range(n):
            base = self._random_ids(self.config.max_len).clone()
            indices = torch.randperm(self.config.max_len, generator=self._rng)[:n_noise]
            noise = torch.randint(
                low=0,
                high=self.config.vocab_size,
                size=(n_noise,),
                generator=self._rng,
                dtype=torch.long,
            )
            base[0, indices] = noise
            samples.append(
                RedTeamSample(
                    prompt_ids=base,
                    label="adversarial",
                    perturbation_type="token_noise",
                )
            )
        return samples

    def generate_dataset(
        self,
        include_types: tuple[str, ...] = ("benign", "prefix", "suffix", "noise"),
    ) -> list[RedTeamSample]:
        all_samples: list[RedTeamSample] = []

        if "benign" in include_types:
            all_samples.extend(self.generate_benign(self.config.n_benign))
        if "prefix" in include_types:
            all_samples.extend(self.generate_prefix_injection(self.config.n_adversarial))
        if "suffix" in include_types:
            all_samples.extend(self.generate_suffix_injection(self.config.n_adversarial))
        if "noise" in include_types:
            all_samples.extend(self.generate_random_token_attack(self.config.n_adversarial))

        random.seed(self.config.seed)
        random.shuffle(all_samples)
        return all_samples

    def to_tensor_batch(self, samples: list[RedTeamSample]) -> Tensor:
        max_len = self.config.max_len
        batch = torch.zeros(len(samples), max_len, dtype=torch.long)
        for i, sample in enumerate(samples):
            ids = sample.prompt_ids.squeeze(0)
            seq_len = min(ids.shape[0], max_len)
            batch[i, :seq_len] = ids[:seq_len]
        return batch
