"""Noise Contrastive Estimation (Gutmann & Hyvärinen, 2010) for large-vocab LM training."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class NCEConfig:
    """Configuration for NCE-based language model loss."""

    n_noise_samples: int = 20
    noise_dist: str = "unigram"          # "unigram" | "uniform"
    noise_exponent: float = 0.75         # word2vec-style smoothing
    temperature: float = 1.0


# ---------------------------------------------------------------------------
# Unigram sampler
# ---------------------------------------------------------------------------

class UnigramSampler:
    """Smoothed unigram noise sampler.

    Computes ``probs = freqs^exponent / sum(freqs^exponent)`` and supports
    multinomial sampling and log-probability queries.
    """

    def __init__(self, token_freqs: Tensor, exponent: float = 0.75) -> None:
        """
        Args:
            token_freqs: ``(V,)`` tensor of token frequency counts.
            exponent: smoothing exponent (0.75 from word2vec).
        """
        smoothed = token_freqs.float().pow(exponent)
        self.probs: Tensor = smoothed / smoothed.sum()

    def sample(self, n: int, seed: int | None = None) -> Tensor:
        """Sample *n* token ids from the smoothed unigram distribution.

        Args:
            n: number of samples to draw.
            seed: optional RNG seed for reproducibility.

        Returns:
            ``LongTensor`` of shape ``(n,)``.
        """
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)
            return torch.multinomial(self.probs, num_samples=n, replacement=True,
                                     generator=generator)
        return torch.multinomial(self.probs, num_samples=n, replacement=True)

    def log_prob(self, token_ids: Tensor) -> Tensor:
        """Return log probability of each token id.

        Args:
            token_ids: arbitrary-shape int64 tensor of token indices.

        Returns:
            Float tensor with the same shape as *token_ids*.
        """
        probs = self.probs.to(token_ids.device)
        return torch.log(probs[token_ids])


# ---------------------------------------------------------------------------
# NCE binary classification loss (functional)
# ---------------------------------------------------------------------------

def nce_loss(
    scores_pos: Tensor,
    scores_neg: Tensor,
    log_noise_pos: Tensor,
    log_noise_neg: Tensor,
    k: int,
) -> Tensor:
    """NCE binary classification loss.

    Trains a classifier that distinguishes real tokens from noise-sampled ones:

    * Positive term: ``log sigmoid(scores_pos - log(k) - log_noise_pos)``
    * Negative term: ``log sigmoid(-(scores_neg - log(k) - log_noise_neg))``

    Args:
        scores_pos: model scores for positive (real) tokens, shape ``(N,)``.
        scores_neg: model scores for negative (noise) tokens, shape ``(M,)``.
        log_noise_pos: log noise probability for positive tokens, shape ``(N,)``.
        log_noise_neg: log noise probability for negative tokens, shape ``(M,)``.
        k: number of noise samples per positive (used for the ``log(k)`` offset).

    Returns:
        Scalar loss tensor: ``-mean(pos_terms) - mean(neg_terms)``.
    """
    log_k = torch.tensor(float(k), device=scores_pos.device).log()

    pos_terms = F.logsigmoid(scores_pos - log_k - log_noise_pos)   # (N,)
    neg_terms = F.logsigmoid(-(scores_neg - log_k - log_noise_neg))  # (M,)

    return -pos_terms.mean() - neg_terms.mean()


# ---------------------------------------------------------------------------
# Sampled softmax (TF-style)
# ---------------------------------------------------------------------------

def sampled_softmax_loss(
    logits: Tensor,
    targets: Tensor,
    sampled_ids: Tensor,
    log_probs_sampled: Tensor,
) -> Tensor:
    """Sampled softmax loss with biased logit correction (TF-style).

    Args:
        logits: ``(B, V)`` full vocabulary logits.
        targets: ``(B,)`` positive token ids.
        sampled_ids: ``(S,)`` sampled noise token ids.
        log_probs_sampled: ``(S,)`` log noise probabilities for *sampled_ids*.

    Returns:
        Scalar cross-entropy loss over the local vocabulary
        (unique union of targets + sampled_ids).
    """
    device = logits.device

    # Build local vocabulary: unique union of targets and sampled_ids
    all_ids = torch.cat([targets, sampled_ids], dim=0)
    local_ids, inverse = torch.unique(all_ids, return_inverse=True)  # (L,)

    # Gather logits for local vocab — shape (B, L)
    local_logits = logits[:, local_ids]  # (B, L)

    # Apply noise correction to sampled positions
    # We need the positions of sampled_ids within local_ids
    # The first len(targets) entries of `inverse` map targets; the rest map sampled_ids
    sampled_local_idx = inverse[targets.shape[0]:]  # (S,) positions in local_ids

    # Build correction vector of shape (L,) — zero for non-sampled tokens
    correction = torch.zeros(local_ids.shape[0], device=device)
    # For tokens that appear multiple times in sampled_ids we take the last correction;
    # all share the same log_probs value per unique id, so scatter is fine.
    correction.scatter_(0, sampled_local_idx, log_probs_sampled)

    # Subtract correction from local logits (broadcast over batch)
    local_logits = local_logits - correction.unsqueeze(0)  # (B, L)

    # Map target ids to their position in local_ids
    target_local_idx = inverse[:targets.shape[0]]  # (B,)

    return F.cross_entropy(local_logits, target_local_idx)


# ---------------------------------------------------------------------------
# NCE language model loss (nn.Module)
# ---------------------------------------------------------------------------

class NCELanguageModelLoss(nn.Module):
    """NCE-based language model loss for efficient large-vocabulary training.

    Replaces the full-vocabulary softmax with a binary NCE classifier,
    sampling a small number of noise tokens per training step.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        config: NCEConfig,
        token_freqs: Tensor | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.config = config

        self.output_proj = nn.Linear(d_model, vocab_size, bias=False)

        if token_freqs is not None:
            self.sampler: UnigramSampler | None = UnigramSampler(
                token_freqs, exponent=config.noise_exponent
            )
        else:
            self.sampler = None

    # ------------------------------------------------------------------
    def forward(self, hidden: Tensor, targets: Tensor) -> tuple[Tensor, dict]:
        """Compute NCE loss.

        Args:
            hidden: ``(B, T, D)`` hidden states.
            targets: ``(B, T)`` ground-truth token ids.

        Returns:
            ``(loss, metrics)`` where *metrics* is a dict containing
            ``"nce_loss"`` (float) and ``"n_noise"`` (int).
        """
        B, T, D = hidden.shape
        device = hidden.device
        k = self.config.n_noise_samples

        # Flatten
        hidden_flat = hidden.reshape(B * T, D)         # (N, D)
        targets_flat = targets.reshape(B * T)           # (N,)
        N = hidden_flat.shape[0]

        # Sample noise ids
        if self.sampler is not None:
            noise_ids = self.sampler.sample(k)
        else:
            noise_ids = torch.randint(0, self.vocab_size, (k,))
        noise_ids = noise_ids.to(device)               # (k,)

        # Full projection — (N, V)
        all_logits = self.output_proj(hidden_flat)
        if self.config.temperature != 1.0:
            all_logits = all_logits / self.config.temperature

        # Positive scores: one per position
        scores_pos = all_logits[torch.arange(N, device=device), targets_flat]  # (N,)

        # Negative scores: (N, k)
        scores_neg = all_logits[:, noise_ids]          # (N, k)
        scores_neg_flat = scores_neg.reshape(-1)       # (N*k,)

        # Noise log-probs
        if self.sampler is not None:
            log_noise_pos = self.sampler.log_prob(targets_flat.cpu()).to(device)     # (N,)
            noise_ids_exp = noise_ids.unsqueeze(0).expand(N, k).reshape(-1)         # (N*k,)
            log_noise_neg = self.sampler.log_prob(noise_ids_exp.cpu()).to(device)    # (N*k,)
        else:
            log_uniform = -torch.log(torch.tensor(float(self.vocab_size), device=device))
            log_noise_pos = torch.full((N,), log_uniform.item(), device=device)
            log_noise_neg = torch.full((N * k,), log_uniform.item(), device=device)

        loss = nce_loss(scores_pos, scores_neg_flat, log_noise_pos, log_noise_neg, k)

        metrics = {
            "nce_loss": loss.item(),
            "n_noise": k,
        }
        return loss, metrics


# ---------------------------------------------------------------------------
# Comparison utility
# ---------------------------------------------------------------------------

def compare_nce_vs_softmax(
    model_scores: Tensor,
    targets: Tensor,
    vocab_size: int,
) -> dict[str, float]:
    """Compare full softmax loss vs a sampled NCE approximation.

    Args:
        model_scores: ``(B, V)`` unnormalised logits.
        targets: ``(B,)`` ground-truth token ids.
        vocab_size: vocabulary size *V*.

    Returns:
        Dict with keys ``"softmax_loss"``, ``"nce_approx_loss"``, and
        ``"relative_error"``.
    """
    # Full softmax cross-entropy
    softmax_loss_val = F.cross_entropy(model_scores, targets).item()

    # NCE approximation: binary classification with uniform noise
    k = 20
    device = model_scores.device
    B = model_scores.shape[0]

    noise_ids = torch.randint(0, vocab_size, (k,), device=device)
    log_uniform = -torch.log(torch.tensor(float(vocab_size), device=device))

    scores_pos = model_scores[torch.arange(B, device=device), targets]          # (B,)
    scores_neg = model_scores[:, noise_ids].reshape(-1)                          # (B*k,)
    log_noise_pos = torch.full((B,), log_uniform.item(), device=device)
    log_noise_neg = torch.full((B * k,), log_uniform.item(), device=device)

    nce_approx = nce_loss(scores_pos, scores_neg, log_noise_pos, log_noise_neg, k).item()

    relative_error = abs(nce_approx - softmax_loss_val) / (abs(softmax_loss_val) + 1e-8)

    return {
        "softmax_loss": softmax_loss_val,
        "nce_approx_loss": nce_approx,
        "relative_error": relative_error,
    }
