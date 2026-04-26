"""Energy-based models (EBM) with contrastive divergence for language modeling.

Implements:
- EBMConfig: configuration dataclass
- compute_energy: negative log-likelihood as energy per sequence
- gibbs_sample_step: discrete MCMC sampling via Gibbs-style token replacement
- ReplayBuffer: persistent contrastive divergence replay buffer
- EBMTrainer: contrastive divergence training loop
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor


@dataclass
class EBMConfig:
    """Configuration for energy-based model training with contrastive divergence."""

    cd_steps: int = 1
    """Number of contrastive divergence MCMC steps."""

    mcmc_step_size: float = 0.1
    """Step size for MCMC transitions (used to scale sampling temperature)."""

    n_negative_samples: int = 4
    """Number of negative samples to generate per training step."""

    energy_margin: float = 0.5
    """Margin for energy regularization term."""

    replay_buffer_size: int = 100
    """Maximum number of sequences to store in the replay buffer."""


def compute_energy(model: object, input_ids: Tensor) -> Tensor:
    """Compute negative log-likelihood energy for each sequence.

    Energy = mean(-log p(token | context)) over token positions per sequence.
    Uses next-token prediction: logits[:, :-1] vs input_ids[:, 1:].

    Args:
        model: AureliusTransformer — callable returning (loss, logits, pkv).
        input_ids: (B, T) token ids.

    Returns:
        Tensor of shape (B,) — energy per sequence (higher = less likely).
    """
    with torch.no_grad():
        _, logits, _ = model(input_ids)  # (B, T, V)

    B, T, V = logits.shape

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
    shift_targets = input_ids[:, 1:].contiguous()  # (B, T-1)

    # Per-token cross-entropy (equivalent to -log_softmax at target token)
    per_token_loss = F.cross_entropy(
        shift_logits.view(-1, V),
        shift_targets.view(-1),
        reduction="none",
    )  # (B*(T-1),)
    per_token_loss = per_token_loss.view(B, T - 1)  # (B, T-1)

    # Mean over token dimension → energy per sequence
    energy = per_token_loss.mean(dim=1)  # (B,)
    return energy


def gibbs_sample_step(
    model: object,
    input_ids: Tensor,
    n_steps: int = 1,
) -> Tensor:
    """Discrete MCMC via Gibbs-style token replacement.

    For each step: randomly pick one position per sequence, replace it with
    a token sampled from the model's conditional distribution at that position.

    Args:
        model: AureliusTransformer — callable returning (loss, logits, pkv).
        input_ids: (B, T) token ids.
        n_steps: number of Gibbs sweep steps.

    Returns:
        Modified input_ids of the same shape (B, T).
    """
    current = input_ids.clone()
    B, T = current.shape

    for _ in range(n_steps):
        with torch.no_grad():
            _, logits, _ = model(current)  # (B, T, V)

        # For each sequence, randomly pick a position to update
        # We use positions 1..T-1 so we always have context (avoid position 0)
        # and use logits from the previous position to predict it
        if T <= 1:
            break

        # Sample replacement positions: for each batch item, pick pos in [1, T-1]
        pos = torch.randint(1, T, (B,), device=current.device)  # (B,)

        # For position p, the logits at position p-1 predict token at p
        for b in range(B):
            p = pos[b].item()
            token_logits = logits[b, p - 1, :]  # (V,)
            sampled_token = torch.multinomial(
                F.softmax(token_logits, dim=-1), num_samples=1
            ).squeeze(0)
            current[b, p] = sampled_token

    return current


class ReplayBuffer:
    """Fixed-size FIFO replay buffer storing negative samples for persistent CD.

    Stores (B, T) token sequences. When capacity is exceeded, oldest entries
    are evicted (deque with maxlen).
    """

    def __init__(self, buffer_size: int) -> None:
        self._buffer: deque[Tensor] = deque()
        self._buffer_size = buffer_size
        self._total_sequences = 0  # track total stored (not deque size)

    def add(self, samples: Tensor) -> None:
        """Add a batch of sequences to the buffer.

        Args:
            samples: (B, T) tensor of token sequences.
        """
        B = samples.shape[0]
        for i in range(B):
            seq = samples[i].detach().cpu()
            self._buffer.append(seq)
            if len(self._buffer) > self._buffer_size:
                self._buffer.popleft()

    def sample(self, n: int) -> Tensor | None:
        """Sample n sequences from the buffer uniformly at random.

        Args:
            n: number of sequences to sample.

        Returns:
            Tensor of shape (n, T) or None if the buffer is empty.
        """
        if len(self._buffer) == 0:
            return None

        buf_list = list(self._buffer)
        indices = torch.randint(0, len(buf_list), (n,))
        seqs = [buf_list[i.item()] for i in indices]
        return torch.stack(seqs, dim=0)  # (n, T)

    def __len__(self) -> int:
        return len(self._buffer)


class EBMTrainer:
    """Contrastive divergence training for energy-based language models.

    Implements the standard CD-k objective:
        loss = mean(E_pos) - mean(E_neg) + regularization

    Regularization follows the hinge/margin form to prevent energy collapse:
        reg = energy_margin * mean((E_pos + E_neg) ** 2)

    Negative samples are drawn from the replay buffer when available; otherwise
    they are generated on-the-fly via Gibbs sampling from positive data.
    """

    def __init__(
        self,
        model: object,
        config: EBMConfig,
        optimizer: object,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.replay_buffer = ReplayBuffer(config.replay_buffer_size)

    def train_step(self, pos_ids: Tensor) -> dict:
        """Run one contrastive divergence training step.

        Args:
            pos_ids: (B, T) positive (real) token sequences.

        Returns:
            Dict with keys: "loss" (float), "energy_pos" (float), "energy_neg" (float).
        """
        device = pos_ids.device

        # --- Get negative samples ---
        buf_sample = self.replay_buffer.sample(self.config.n_negative_samples)
        if buf_sample is not None:
            neg_ids = buf_sample.to(device)
            # Optionally refine buffer samples via one Gibbs step
            neg_ids = gibbs_sample_step(self.model, neg_ids, n_steps=self.config.cd_steps)
        else:
            # Cold start: initialize negatives from positives and run Gibbs
            # Tile positive samples to get n_negative_samples worth of negatives
            B = pos_ids.shape[0]
            n_neg = self.config.n_negative_samples
            repeat_times = (n_neg + B - 1) // B  # ceiling div
            neg_ids = pos_ids.repeat(repeat_times, 1)[:n_neg]
            neg_ids = gibbs_sample_step(self.model, neg_ids, n_steps=self.config.cd_steps)

        # Store new negatives back to replay buffer
        self.replay_buffer.add(neg_ids.detach())

        # --- Compute energies (with grad for positive, no-grad handled inside) ---
        # We need gradients through energy for the model parameters, so recompute
        # with grad enabled for positive sequences
        _, logits_pos, _ = self.model(pos_ids)  # (B, T, V)
        B_pos, T_pos, V = logits_pos.shape
        shift_logits_pos = logits_pos[:, :-1, :].contiguous()
        shift_targets_pos = pos_ids[:, 1:].contiguous()
        per_token_pos = F.cross_entropy(
            shift_logits_pos.view(-1, V),
            shift_targets_pos.view(-1),
            reduction="none",
        ).view(B_pos, T_pos - 1)
        E_pos = per_token_pos.mean(dim=1)  # (B_pos,)

        with torch.no_grad():
            E_neg_val = compute_energy(self.model, neg_ids)  # (n_neg,)

        # CD loss: push positive energy up, negative energy down
        # + regularization to prevent energy from collapsing
        mean_E_pos = E_pos.mean()
        mean_E_neg = E_neg_val.mean()

        cd_loss = mean_E_pos - mean_E_neg
        reg = self.config.energy_margin * (mean_E_pos + mean_E_neg) ** 2
        loss = cd_loss + reg

        # --- Backward and optimizer step ---
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "energy_pos": mean_E_pos.item(),
            "energy_neg": mean_E_neg.item(),
        }
