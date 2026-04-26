"""Noise-aware training with label smoothing, token noise injection, and mixup."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NoiseConfig:
    """Configuration for noise-aware training."""

    label_smoothing: float = 0.1
    token_noise_prob: float = 0.05
    mixup_alpha: float = 0.2
    noise_type: str = "uniform"  # "uniform" | "gaussian" | "none"


def label_smoothed_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    smoothing: float,
    ignore_index: int = -100,
) -> torch.Tensor:
    """Cross-entropy loss with label smoothing.

    Loss = (1 - smoothing) * CE(logits, targets) + smoothing * mean(-log_softmax(logits))

    Args:
        logits: (batch * seq_len, vocab_size) or (batch, seq_len, vocab_size)
        targets: (batch * seq_len,) or (batch, seq_len)  — integer class labels
        smoothing: Label smoothing coefficient in [0, 1].
        ignore_index: Positions with this label are excluded from the loss.

    Returns:
        Scalar loss tensor.
    """
    # Flatten to 2-D / 1-D
    if logits.dim() == 3:
        B, S, V = logits.shape
        logits = logits.view(B * S, V)
        targets = targets.view(B * S)

    logits.size(-1)

    # Build valid mask (positions that are NOT ignore_index)
    mask = targets != ignore_index  # (N,)
    if not mask.any():
        return logits.sum() * 0.0  # keep graph alive, return 0

    logits_valid = logits[mask]  # (M, V)
    targets_valid = targets[mask]  # (M,)

    # Standard cross-entropy part
    ce = F.cross_entropy(logits_valid, targets_valid, reduction="mean")

    if smoothing == 0.0:
        return ce

    # Smoothing part: mean over all classes of -log_softmax averaged over valid tokens
    log_probs = F.log_softmax(logits_valid, dim=-1)  # (M, V)
    smooth_loss = -log_probs.mean()  # scalar

    return (1.0 - smoothing) * ce + smoothing * smooth_loss


def inject_token_noise(
    input_ids: torch.Tensor,
    noise_prob: float,
    vocab_size: int,
    rng: torch.Generator | None = None,
) -> torch.Tensor:
    """Randomly replace tokens with uniform random tokens.

    Args:
        input_ids: (batch, seq_len) integer token tensor.
        noise_prob: Probability of replacing each token.
        vocab_size: Size of vocabulary (replacement tokens drawn from [0, vocab_size)).
        rng: Optional torch.Generator for reproducibility.

    Returns:
        Modified input_ids (same shape, same dtype, on same device).
    """
    if noise_prob <= 0.0:
        return input_ids.clone()

    ids = input_ids.clone()
    device = ids.device

    # Generate noise mask
    mask = torch.bernoulli(
        torch.full(ids.shape, noise_prob, dtype=torch.float32, device=device),
        generator=rng,
    ).bool()

    # Generate replacement tokens
    random_tokens = torch.randint(
        0,
        vocab_size,
        ids.shape,
        dtype=ids.dtype,
        device=device,
        generator=rng,
    )

    ids[mask] = random_tokens[mask]
    return ids


def mixup_embeddings(
    embed_fn,
    ids_a: torch.Tensor,
    ids_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    """Compute a convex combination of two embedding sequences.

    Args:
        embed_fn: Callable mapping (batch, seq_len) -> (batch, seq_len, d_model).
        ids_a: First set of token ids.
        ids_b: Second set of token ids.
        lam: Mixup coefficient in [0, 1]. Output = lam * embed(ids_a) + (1-lam) * embed(ids_b).

    Returns:
        Mixed embedding tensor (batch, seq_len, d_model).
    """
    emb_a = embed_fn(ids_a)
    emb_b = embed_fn(ids_b)
    return lam * emb_a + (1.0 - lam) * emb_b


def sample_mixup_lambda(alpha: float) -> float:
    """Sample a mixup coefficient from Beta(alpha, alpha).

    Args:
        alpha: Beta distribution concentration parameter. If 0, returns 1.0.

    Returns:
        Float lambda value in (0, 1] (or 1.0 when alpha == 0).
    """
    if alpha == 0.0:
        return 1.0
    beta = torch.distributions.Beta(
        torch.tensor(alpha, dtype=torch.float32),
        torch.tensor(alpha, dtype=torch.float32),
    )
    return float(beta.sample().item())


class NoiseAwareTrainer:
    """Trainer that applies configurable input noise during training."""

    def __init__(
        self,
        model: nn.Module,
        config: NoiseConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = optimizer

    def train_step(self, input_ids: torch.Tensor) -> dict:
        """Single training step with token noise injection.

        Args:
            input_ids: (batch, seq_len) integer token tensor.

        Returns:
            dict with keys "loss" (float) and "noise_type" (str).
        """
        self.model.train()
        self.optimizer.zero_grad()

        vocab_size = self.model.config.vocab_size
        noisy_ids = inject_token_noise(
            input_ids,
            noise_prob=self.config.token_noise_prob,
            vocab_size=vocab_size,
        )

        # Labels are the original (un-noised) next tokens
        labels = input_ids

        _, logits, _ = self.model(noisy_ids)

        # Shift for next-token prediction
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        loss = label_smoothed_loss(
            shift_logits,
            shift_labels,
            smoothing=self.config.label_smoothing,
            ignore_index=-100,
        )

        loss.backward()
        self.optimizer.step()

        return {"loss": float(loss.item()), "noise_type": self.config.noise_type}

    def mixup_train_step(
        self,
        input_ids_a: torch.Tensor,
        input_ids_b: torch.Tensor,
    ) -> dict:
        """Training step using embedding-space mixup.

        Embeds both input sequences, mixes them with a sampled lambda, then
        passes the mixed embeddings through transformer layers directly.

        Args:
            input_ids_a: (batch, seq_len) first sequence.
            input_ids_b: (batch, seq_len) second sequence.

        Returns:
            dict with keys "loss" (float) and "lam" (float).
        """
        self.model.train()
        self.optimizer.zero_grad()

        lam = sample_mixup_lambda(self.config.mixup_alpha)

        # Mixed embeddings
        mixed_emb = mixup_embeddings(self.model.embed, input_ids_a, input_ids_b, lam)

        # Forward through transformer layers manually (skip embed step)
        B, S, _ = mixed_emb.shape
        assert S <= self.model.config.max_seq_len  # noqa: S101

        x = mixed_emb
        freqs_cis = self.model.freqs_cis[:S]

        for layer in self.model.layers:
            x, _ = layer(x, freqs_cis, None, None)

        x = self.model.norm(x)
        logits = self.model.lm_head(x)

        # Mixed loss: lam * loss_a + (1 - lam) * loss_b
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels_a = input_ids_a[:, 1:].contiguous()
        shift_labels_b = input_ids_b[:, 1:].contiguous()

        loss_a = label_smoothed_loss(
            shift_logits,
            shift_labels_a,
            smoothing=self.config.label_smoothing,
            ignore_index=-100,
        )
        loss_b = label_smoothed_loss(
            shift_logits,
            shift_labels_b,
            smoothing=self.config.label_smoothing,
            ignore_index=-100,
        )
        loss = lam * loss_a + (1.0 - lam) * loss_b

        loss.backward()
        self.optimizer.step()

        return {"loss": float(loss.item()), "lam": lam}
