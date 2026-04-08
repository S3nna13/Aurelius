"""Importance-weighted sampling for curriculum training."""

import torch
from torch.utils.data import WeightedRandomSampler
from dataclasses import dataclass


@dataclass
class ImportanceSamplerConfig:
    temperature: float = 1.0       # weight = perplexity^(1/temperature)
    # T < 1: sharper (focus harder)
    # T > 1: flatter (more uniform)
    # T -> inf: uniform
    min_weight: float = 0.01       # floor weight to prevent starvation
    ema_alpha: float = 0.3         # smoothing for online weight updates


class ImportanceWeightedSampler:
    """Assigns sampling weights based on per-example difficulty.

    Usage:
        sampler = ImportanceWeightedSampler(dataset, cfg)

        # Initialize with uniform or pre-computed weights
        sampler.set_weights(perplexities)  # high perplexity = upweighted

        # Get a PyTorch sampler for DataLoader
        torch_sampler = sampler.get_sampler(num_samples=len(dataset))
        loader = DataLoader(dataset, batch_sampler=None, sampler=torch_sampler)
    """

    def __init__(
        self,
        n_examples: int,
        cfg: ImportanceSamplerConfig | None = None,
    ):
        self.n_examples = n_examples
        self.cfg = cfg or ImportanceSamplerConfig()
        self._weights = torch.ones(n_examples)
        self._perplexities = torch.ones(n_examples)

    def set_weights(self, perplexities: torch.Tensor) -> None:
        """Set weights from perplexity scores (higher ppl = higher weight).

        weight_i = max(ppl_i^(1/T), min_weight)
        Then normalized to sum to n_examples (preserves expected batch size).
        """
        ppl = perplexities.float().clamp(min=1.0)
        self._perplexities = ppl.clone()

        # weight = ppl^(1/T)
        exponent = 1.0 / max(self.cfg.temperature, 1e-8)
        raw = ppl ** exponent

        # Apply min_weight floor
        raw = raw.clamp(min=self.cfg.min_weight)

        # Normalize to sum to n_examples
        self._weights = raw / raw.sum() * self.n_examples

    def update_weight(self, idx: int, new_perplexity: float) -> None:
        """Update a single example's perplexity using EMA."""
        old_ppl = self._perplexities[idx].item()
        new_ppl_ema = self.cfg.ema_alpha * new_perplexity + (1 - self.cfg.ema_alpha) * old_ppl
        updated_ppls = self._perplexities.clone()
        updated_ppls[idx] = new_ppl_ema
        self.set_weights(updated_ppls)

    def get_sampler(self, num_samples: int | None = None) -> WeightedRandomSampler:
        """Return a WeightedRandomSampler for use in DataLoader."""
        n = num_samples or self.n_examples
        return WeightedRandomSampler(
            weights=self._weights.tolist(),
            num_samples=n,
            replacement=True,
        )

    @property
    def weights(self) -> torch.Tensor:
        return self._weights.clone()

    @property
    def perplexities(self) -> torch.Tensor:
        return self._perplexities.clone()


class DynamicCurriculumSampler:
    """Schedules importance sampling temperature over training steps.

    Early training: high temperature (near-uniform) -- explore all data.
    Late training: low temperature (focused) -- concentrate on hard examples.
    """

    def __init__(
        self,
        n_examples: int,
        total_steps: int,
        start_temperature: float = 10.0,   # near-uniform early
        end_temperature: float = 1.0,      # focused late
        cfg: ImportanceSamplerConfig | None = None,
    ):
        self.total_steps = total_steps
        self.start_temp = start_temperature
        self.end_temp = end_temperature
        self._step = 0
        base_cfg = cfg or ImportanceSamplerConfig()
        self.sampler = ImportanceWeightedSampler(n_examples, base_cfg)

    def get_temperature(self, step: int) -> float:
        """Linear decay from start_temperature to end_temperature."""
        progress = min(step / max(self.total_steps, 1), 1.0)
        return self.start_temp + (self.end_temp - self.start_temp) * progress

    def step(self, perplexities: torch.Tensor) -> WeightedRandomSampler:
        """Update weights with current temperature and return sampler."""
        temp = self.get_temperature(self._step)
        cfg = ImportanceSamplerConfig(
            temperature=temp,
            min_weight=self.sampler.cfg.min_weight,
        )
        temp_sampler = ImportanceWeightedSampler(self.sampler.n_examples, cfg)
        temp_sampler.set_weights(perplexities)
        self._step += 1
        return temp_sampler.get_sampler()
