"""Inference for Discrete Diffusion Language Models.

Implements sampling from masked discrete diffusion models (MDLM/D3PM style).
The generative process starts from all-[MASK] and iteratively unmasks tokens.

References:
    Austin et al. 2021 (D3PM) — https://arxiv.org/abs/2107.03006
    Shi et al. 2024 (MDLM) — https://arxiv.org/abs/2406.07524
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import torch
import torch.nn.functional as F
from torch import Tensor


class MaskNoiseSchedule:
    """Controls the masking schedule for discrete diffusion.

    Uses a linear masking schedule where the sequence is fully masked at t=0
    and fully unmasked at t=n_steps.
    """

    def __init__(self, n_steps: int = 100, mask_token_id: int = 0) -> None:
        self.n_steps = n_steps
        self.mask_token_id = mask_token_id

    def mask_rate(self, t: int) -> float:
        """Return fraction of tokens masked at timestep t.

        Linear schedule: fully masked at t=0, fully unmasked at t=n_steps.
        """
        return 1.0 - t / self.n_steps

    def unmask_rate(self, t: int) -> float:  # noqa: ARG002
        """Rate at which tokens get unmasked from t to t-1."""
        return 1.0 / self.n_steps

    def apply_mask(
        self,
        token_ids: Tensor,
        t: int,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Mask each token independently with probability mask_rate(t).

        Args:
            token_ids: ``(B, T)`` long tensor of token ids.
            t: Current timestep.
            generator: Optional random number generator for reproducibility.

        Returns:
            ``(B, T)`` tensor with some tokens replaced by ``mask_token_id``.
        """
        rate = self.mask_rate(t)
        if rate <= 0.0:
            return token_ids.clone()
        if rate >= 1.0:
            return torch.full_like(token_ids, self.mask_token_id)

        noise = torch.rand(token_ids.shape, generator=generator, dtype=torch.float32)
        mask = noise < rate
        result = token_ids.clone()
        result[mask] = self.mask_token_id
        return result


class MDLMSampler:
    """Ancestral sampler for masked diffusion LM (MDLM / D3PM style).

    Starts from a fully masked sequence and iteratively unmasks tokens by
    querying a denoising model at each timestep.
    """

    def __init__(
        self,
        model_fn: Callable[[Tensor, int], Tensor],
        schedule: MaskNoiseSchedule,
        vocab_size: int,
    ) -> None:
        """
        Args:
            model_fn: ``(noisy_ids: LongTensor(B, T), t: int) -> logits: (B, T, V)``
            schedule: Masking noise schedule.
            vocab_size: Size of the token vocabulary.
        """
        self.model_fn = model_fn
        self.schedule = schedule
        self.vocab_size = vocab_size

    def denoise_step(self, noisy_ids: Tensor, t: int) -> Tensor:
        """Single denoising step from timestep t to t-1.

        For each MASKED position: sample a new token from softmax(logits).
        For each UNMASKED position: keep the existing token unchanged.

        Args:
            noisy_ids: ``(B, T)`` long tensor with some mask tokens.
            t: Current timestep (will step to t-1).

        Returns:
            ``(B, T)`` tensor with fewer mask tokens.
        """
        logits = self.model_fn(noisy_ids, t)  # (B, T, V)

        # Sample from the predicted distribution at each position
        B, T, V = logits.shape
        flat_logits = logits.reshape(B * T, V)
        flat_samples = torch.multinomial(
            F.softmax(flat_logits, dim=-1), num_samples=1
        ).squeeze(-1)  # (B*T,)
        sampled = flat_samples.reshape(B, T)

        # Only replace MASKED positions; keep unmasked tokens intact
        is_masked = noisy_ids == self.schedule.mask_token_id
        result = noisy_ids.clone()
        result[is_masked] = sampled[is_masked]
        return result

    def sample(
        self,
        batch_size: int,
        seq_len: int,
        n_steps: int | None = None,
    ) -> Tensor:
        """Generate sequences via iterative denoising.

        Starts from an all-mask sequence and iteratively denoises from
        t=n_steps down to t=1.

        Args:
            batch_size: Number of sequences to generate.
            seq_len: Length of each generated sequence.
            n_steps: Number of denoising steps (defaults to ``schedule.n_steps``).

        Returns:
            ``(batch_size, seq_len)`` fully denoised long tensor.
        """
        if n_steps is None:
            n_steps = self.schedule.n_steps

        x = torch.full(
            (batch_size, seq_len),
            self.schedule.mask_token_id,
            dtype=torch.long,
        )

        for t in range(n_steps, 0, -1):
            x = self.denoise_step(x, t)

        return x


class ContinuousTimeDiffusionSampler:
    """MDLM-style sampler with continuous time (arbitrary SNR schedule).

    Operates in continuous time t ∈ (0, 1) where t=1 is fully noisy and
    t=0 is fully clean.
    """

    def __init__(
        self,
        model_fn: Callable[[Tensor, float], Tensor],
        vocab_size: int,
        mask_token_id: int = 0,
    ) -> None:
        """
        Args:
            model_fn: ``(noisy_ids: LongTensor(B, T), t: float) -> logits: (B, T, V)``
            vocab_size: Size of the token vocabulary.
            mask_token_id: Token id used for masking.
        """
        self.model_fn = model_fn
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id

    def log_snr(self, t: float) -> float:
        """Log signal-to-noise ratio.

        Returns ``-log(t / (1 - t + 1e-6))``.  At t=0.5 this is 0 (balanced).
        """
        return -math.log(t / (1.0 - t + 1e-6))

    def sample_timesteps(self, n: int) -> Tensor:
        """Sample n timesteps uniformly from (0, 1).

        Args:
            n: Number of timestep samples.

        Returns:
            ``(n,)`` float tensor of uniform samples in (0, 1).
        """
        return torch.rand(n)

    def denoise(self, noisy_ids: Tensor, t: float, t_prev: float) -> Tensor:
        """Partially denoise from continuous timestep t to t_prev.

        For each masked position, unmask it with probability
        ``(t - t_prev) / t``.

        Args:
            noisy_ids: ``(B, T)`` long tensor with some mask tokens.
            t: Current (noisier) timestep, t > t_prev.
            t_prev: Target (cleaner) timestep.

        Returns:
            ``(B, T)`` partially denoised tensor.
        """
        logits = self.model_fn(noisy_ids, t)  # (B, T, V)

        B, T, V = logits.shape
        flat_logits = logits.reshape(B * T, V)
        flat_samples = torch.multinomial(
            F.softmax(flat_logits, dim=-1), num_samples=1
        ).squeeze(-1)
        sampled = flat_samples.reshape(B, T)

        # Unmask probability for each currently masked position
        unmask_prob = (t - t_prev) / (t + 1e-8)
        is_masked = noisy_ids == self.mask_token_id

        # Draw Bernoulli decisions for each masked position
        noise = torch.rand(noisy_ids.shape)
        do_unmask = (noise < unmask_prob) & is_masked

        result = noisy_ids.clone()
        result[do_unmask] = sampled[do_unmask]
        return result


class DiffusionLMQualityMetrics:
    """Measures quality of diffusion LM samples."""

    def __init__(self, mask_token_id: int = 0) -> None:
        self.mask_token_id = mask_token_id

    def mask_fraction(self, sequences: Tensor) -> float:
        """Return the fraction of tokens still masked in sequences.

        Args:
            sequences: ``(B, T)`` long tensor.

        Returns:
            Float in [0, 1].
        """
        total = sequences.numel()
        if total == 0:
            return 0.0
        masked = (sequences == self.mask_token_id).sum().item()
        return float(masked) / float(total)

    def unique_fraction(self, sequences: Tensor) -> float:
        """Return the fraction of unique sequences in the batch.

        Args:
            sequences: ``(B, T)`` long tensor.

        Returns:
            Float in [0, 1].
        """
        B = sequences.shape[0]
        if B == 0:
            return 0.0
        unique_rows = {tuple(row.tolist()) for row in sequences}
        return float(len(unique_rows)) / float(B)

    def token_entropy(self, sequences: Tensor, vocab_size: int) -> float:
        """Marginal entropy of the token distribution across all positions.

        Computes empirical token frequencies across all positions and sequences,
        then returns the Shannon entropy.

        Args:
            sequences: ``(B, T)`` long tensor.
            vocab_size: Total vocabulary size.

        Returns:
            Non-negative entropy value (in nats).
        """
        flat = sequences.reshape(-1)
        counts = torch.zeros(vocab_size, dtype=torch.float32)
        for tok_id in flat.tolist():
            if 0 <= tok_id < vocab_size:
                counts[tok_id] += 1.0

        total = counts.sum().item()
        if total == 0:
            return 0.0

        probs = counts / total
        # Avoid log(0) by filtering zero-probability entries
        nonzero = probs[probs > 0]
        entropy = -(nonzero * torch.log(nonzero)).sum().item()
        return float(entropy)
