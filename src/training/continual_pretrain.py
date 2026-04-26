"""Continual pre-training with experience replay to prevent catastrophic forgetting.

Implements:
- ReplayConfig: configuration for the experience replay buffer
- ContinualPretrainConfig: configuration for continual pre-training
- ExperienceReplayBuffer: reservoir-sampling buffer for token sequences
- DomainBatch: data container for a single domain batch
- ContinualPretrainer: trainer with replay-based forgetting prevention
- compute_domain_mixing_weights: softmax weights from per-domain losses

References:
    Rolnick et al. 2019 (Experience Replay for Continual Learning)
    Vitter 1985 (Random Sampling with a Reservoir)
"""

from __future__ import annotations

import math
import random
from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass
class ReplayConfig:
    """Configuration for the experience replay buffer."""

    buffer_size: int = 1000  # maximum number of sequences stored
    replay_ratio: float = 0.3  # fraction of batch drawn from replay
    reservoir_sampling: bool = True  # use reservoir sampling when buffer is full


@dataclass
class ContinualPretrainConfig:
    """Configuration for continual pre-training."""

    learning_rate: float = 1e-4
    max_seq_len: int = 512
    warmup_steps: int = 100
    replay: ReplayConfig = field(default_factory=ReplayConfig)
    domain_weights: dict[str, float] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Experience replay buffer
# ---------------------------------------------------------------------------


class ExperienceReplayBuffer:
    """Fixed-size replay buffer for 1-D token sequences.

    Supports reservoir sampling (Vitter 1985) to maintain a uniform sample
    from the stream of all sequences seen so far.

    Args:
        config: ReplayConfig controlling capacity and sampling strategy.
    """

    def __init__(self, config: ReplayConfig) -> None:
        self.config = config
        self._buffer: list[Tensor] = []
        self._total_seen: int = 0

    # ------------------------------------------------------------------
    def add(self, tokens: Tensor) -> None:
        """Add a 1-D token sequence to the buffer.

        If the buffer is not yet full, the sequence is appended directly.
        Once full and reservoir_sampling is True, the sequence replaces a
        random existing entry with probability buffer_size / total_seen
        (Knuth / Vitter Algorithm R). If reservoir_sampling is False, the
        oldest entry is overwritten in ring-buffer fashion.
        """
        self._total_seen += 1
        if len(self._buffer) < self.config.buffer_size:
            self._buffer.append(tokens)
        elif self.config.reservoir_sampling:
            # Replace slot j with probability buffer_size / total_seen
            j = random.randint(0, self._total_seen - 1)
            if j < self.config.buffer_size:
                self._buffer[j] = tokens
        else:
            # Ring buffer: overwrite oldest slot
            idx = (self._total_seen - 1) % self.config.buffer_size
            self._buffer[idx] = tokens

    # ------------------------------------------------------------------
    def sample(self, n: int) -> list[Tensor]:
        """Sample n sequences uniformly at random (with replacement if n > len).

        Returns an empty list if the buffer is empty.
        """
        if len(self._buffer) == 0:
            return []
        if n <= len(self._buffer):
            return random.sample(self._buffer, n)
        # with replacement when requesting more than available
        return random.choices(self._buffer, k=n)

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self._buffer)


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------


@dataclass
class DomainBatch:
    """A batch of token sequences from a single domain."""

    tokens: Tensor  # shape (B, T)
    domain: str
    weights: Tensor | None = None


# ---------------------------------------------------------------------------
# Continual pre-trainer
# ---------------------------------------------------------------------------


class ContinualPretrainer:
    """Trainer that mixes new-domain data with replayed old-domain data.

    Uses an ExperienceReplayBuffer to retain a compressed memory of
    previously seen sequences and interleaves them with each new batch
    to mitigate catastrophic forgetting.

    Args:
        model:            A model whose forward returns (loss, logits, pkv).
        optimizer:        Any PyTorch optimizer.
        config:           ContinualPretrainConfig.
        tokenizer_encode: Callable that encodes a string to a list of ints
                          (stored for API completeness, used by callers).
    """

    def __init__(
        self,
        model,
        optimizer,
        config: ContinualPretrainConfig,
        tokenizer_encode: Callable,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.config = config
        self.tokenizer_encode = tokenizer_encode
        self.replay_buffer = ExperienceReplayBuffer(config.replay)

    # ------------------------------------------------------------------
    def _compute_loss(self, input_ids: Tensor) -> Tensor:
        """Cross-entropy next-token prediction loss with pad (0) ignored.

        Args:
            input_ids: (B, T) token tensor.

        Returns:
            Scalar loss tensor.
        """
        loss, logits, _pkv = self.model(input_ids)

        if loss is not None:
            return loss

        # Manually compute shifted cross-entropy ignoring pad token (id=0).
        # logits: (B, T, V)  →  shift so position t predicts token t+1.
        shift_logits = logits[:, :-1, :].contiguous()  # (B, T-1, V)
        shift_labels = input_ids[:, 1:].contiguous()  # (B, T-1)

        B, T, V = shift_logits.shape
        loss = F.cross_entropy(
            shift_logits.view(B * T, V),
            shift_labels.view(B * T),
            ignore_index=0,
        )
        return loss

    # ------------------------------------------------------------------
    def train_step(
        self,
        new_tokens: list[Tensor],
        domain: str = "new",
    ) -> dict:
        """One training step mixing new sequences with replay sequences.

        Steps:
        1. Add every new sequence to the replay buffer.
        2. Determine how many replay sequences to mix in.
        3. Sample from replay buffer (if non-empty and replay_n > 0).
        4. Pad all sequences to the maximum length in the batch, then stack.
        5. Forward pass, compute loss, backward, optimizer step.

        Args:
            new_tokens: List of 1-D token tensors (variable length OK).
            domain:     Domain label string for logging.

        Returns:
            dict with keys: "loss", "domain", "n_new", "n_replay".
        """
        # Step 1: add new sequences to replay buffer
        for seq in new_tokens:
            self.replay_buffer.add(seq)

        # Step 2: determine replay count
        replay_n = int(len(new_tokens) * self.config.replay.replay_ratio)

        # Step 3: sample replay sequences
        replay_seqs: list[Tensor] = []
        if len(self.replay_buffer) > 0 and replay_n > 0:
            replay_seqs = self.replay_buffer.sample(replay_n)

        # Step 4: combine and pad to common length
        all_seqs = list(new_tokens) + replay_seqs
        max_len = max(s.shape[0] for s in all_seqs)

        padded = []
        for s in all_seqs:
            pad_amount = max_len - s.shape[0]
            if pad_amount > 0:
                s = F.pad(s, (0, pad_amount), value=0)
            padded.append(s)

        input_ids = torch.stack(padded)  # (B_total, max_len)

        # Step 5: forward, loss, backward, step
        self.model.train()
        self.optimizer.zero_grad()
        loss = self._compute_loss(input_ids)
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "domain": domain,
            "n_new": len(new_tokens),
            "n_replay": len(replay_seqs),
        }

    # ------------------------------------------------------------------
    def evaluate_forgetting(
        self,
        old_sequences: list[Tensor],
        new_sequences: list[Tensor],
    ) -> dict:
        """Compute loss on old and new data to quantify catastrophic forgetting.

        forgetting = old_loss - new_loss
        Positive values indicate the model has forgotten old knowledge.

        Args:
            old_sequences: List of 1-D tensors from a previously trained domain.
            new_sequences: List of 1-D tensors from the new domain.

        Returns:
            dict with keys: "old_loss", "new_loss", "forgetting".
        """
        self.model.train(False)

        def _batch_loss(seqs: list[Tensor]) -> float:
            max_len = max(s.shape[0] for s in seqs)
            padded = []
            for s in seqs:
                pad_amount = max_len - s.shape[0]
                if pad_amount > 0:
                    s = F.pad(s, (0, pad_amount), value=0)
                padded.append(s)
            input_ids = torch.stack(padded)
            with torch.no_grad():
                loss = self._compute_loss(input_ids)
            return loss.item()

        old_loss = _batch_loss(old_sequences)
        new_loss = _batch_loss(new_sequences)
        forgetting = old_loss - new_loss

        return {
            "old_loss": old_loss,
            "new_loss": new_loss,
            "forgetting": forgetting,
        }


# ---------------------------------------------------------------------------
# Domain mixing weights
# ---------------------------------------------------------------------------


def compute_domain_mixing_weights(
    domain_losses: dict[str, float],
    temperature: float = 1.0,
) -> dict[str, float]:
    """Compute softmax mixing weights from per-domain losses.

    Domains with higher loss receive higher weight so training focuses
    on harder / less-learned domains.

    Args:
        domain_losses: Mapping of domain name to current loss value.
        temperature:   Softmax temperature (higher gives more uniform weights).

    Returns:
        Mapping of domain name to mixing weight (values sum to 1.0).
    """
    if not domain_losses:
        return {}

    domains = list(domain_losses.keys())
    losses = [domain_losses[d] for d in domains]

    # Scale by temperature before softmax
    scaled = [loss_val / temperature for loss_val in losses]
    max_scaled = max(scaled)  # numerical stability shift
    exp_vals = [math.exp(s - max_scaled) for s in scaled]
    total = sum(exp_vals)

    return {d: e / total for d, e in zip(domains, exp_vals)}
