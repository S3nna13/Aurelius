"""
energy_based_model.py

Energy-Based Models (EBM) for sequence scoring.
Assigns scalar energy to sequences; trained with contrastive divergence
and noise contrastive estimation (NCE).

Convention: lower energy  →  more likely sequence.
"""

import math
from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class EBMConfig:
    d_model: int = 32
    vocab_size: int = 64
    n_layers: int = 2
    n_mcmc_steps: int = 5
    step_size: float = 0.1
    n_candidates: int = 4
    corrupt_frac: float = 0.15
    lr: float = 1e-4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

class _SinusoidalPositionEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float)
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # shape: [1, max_len, d_model]
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, d_model]
        return x + self.pe[:, : x.size(1)]


class _TransformerEncoderBlock(nn.Module):
    """Single transformer encoder layer (self-attention + FFN)."""

    def __init__(self, d_model: int, n_heads: int = 4, ffn_mult: int = 4) -> None:
        super().__init__()
        # Ensure n_heads divides d_model
        while d_model % n_heads != 0 and n_heads > 1:
            n_heads -= 1
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_mult * d_model),
            nn.GELU(),
            nn.Linear(ffn_mult * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        # FFN with residual
        x = self.norm2(x + self.ffn(x))
        return x


# ---------------------------------------------------------------------------
# SequenceEnergyFunction
# ---------------------------------------------------------------------------

class SequenceEnergyFunction(nn.Module):
    """
    Maps a token sequence  [B, T]  to a scalar energy  [B].

    Architecture:
        Embedding → sinusoidal PE → N transformer encoder layers
        → mean-pool over T → MLP → scalar
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = _SinusoidalPositionEncoding(d_model)

        self.encoder_layers = nn.ModuleList(
            [_TransformerEncoderBlock(d_model) for _ in range(n_layers)]
        )

        # MLP head: d_model → d_model//2 → 1
        hidden = max(d_model // 2, 1)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, hidden),
            nn.GELU(),
            nn.Linear(hidden, 1),
        )

    def _encode(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Encode token ids to a pooled representation.

        Args:
            input_ids: [B, T]

        Returns:
            [B, d_model]
        """
        x = self.embedding(input_ids)          # [B, T, d_model]
        x = self.pos_enc(x)                    # [B, T, d_model]
        for layer in self.encoder_layers:
            x = layer(x)                       # [B, T, d_model]
        pooled = x.mean(dim=1)                 # [B, d_model]
        return pooled

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Return scalar energy for each sequence.

        Args:
            input_ids: [B, T]

        Returns:
            energy: [B]   (lower = more likely)
        """
        pooled = self._encode(input_ids)       # [B, d_model]
        energy = self.mlp(pooled).squeeze(-1)  # [B]
        return energy

    def score(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Unnormalized log probability = -energy.

        Args:
            input_ids: [B, T]

        Returns:
            log_prob: [B]
        """
        return -self.forward(input_ids)


# ---------------------------------------------------------------------------
# NegativeSampler
# ---------------------------------------------------------------------------

class NegativeSampler:
    """Utility class for generating negative (corrupted) sequences."""

    # ------------------------------------------------------------------
    # random_corrupt
    # ------------------------------------------------------------------

    @staticmethod
    def random_corrupt(
        input_ids: torch.Tensor,
        corrupt_frac: float = 0.15,
    ) -> torch.Tensor:
        """Randomly replace *corrupt_frac* fraction of tokens with random tokens.

        Args:
            input_ids:    [B, T]  integer token ids
            corrupt_frac: fraction of positions to corrupt (0 < frac < 1)

        Returns:
            corrupted: [B, T]
        """
        B, T = input_ids.shape
        # Infer vocab size from the max token id present (+1) — used as upper
        # bound for random replacement.  We need at least 1 token id.
        vocab_size = int(input_ids.max().item()) + 1
        # Ensure at least vocab_size=2 so randint is valid
        vocab_size = max(vocab_size, 2)

        corrupted = input_ids.clone()
        # Sample a flat mask with the desired fraction
        mask = torch.rand(B, T, device=input_ids.device) < corrupt_frac  # [B, T] bool
        # Random replacement tokens
        random_tokens = torch.randint(
            0, vocab_size, (B, T), device=input_ids.device
        )
        corrupted[mask] = random_tokens[mask]
        return corrupted

    # ------------------------------------------------------------------
    # masked_corrupt  (BERT MLM style)
    # ------------------------------------------------------------------

    @staticmethod
    def masked_corrupt(
        input_ids: torch.Tensor,
        mask_token_id: int = 0,
    ) -> tuple:
        """Replace ~15 % of tokens with *mask_token_id* (BERT MLM style).

        Args:
            input_ids:    [B, T]
            mask_token_id: token id used as the MASK token

        Returns:
            (masked_ids, mask)  where mask is a bool [B, T] True where masked
        """
        B, T = input_ids.shape
        mask = torch.rand(B, T, device=input_ids.device) < 0.15
        masked_ids = input_ids.clone()
        masked_ids[mask] = mask_token_id
        return masked_ids, mask

    # ------------------------------------------------------------------
    # proposal_sample
    # ------------------------------------------------------------------

    @staticmethod
    def proposal_sample(
        lm_model: nn.Module,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Use an LM to generate alternative tokens at every position.

        The LM must accept integer token ids [B, T] and return logits [B, T, V].
        Each position is replaced with a sample drawn from the LM distribution
        at that position.

        Args:
            lm_model:  nn.Module with forward(input_ids) → logits [B, T, V]
            input_ids: [B, T]

        Returns:
            sampled: [B, T]
        """
        with torch.no_grad():
            logits = lm_model(input_ids)           # [B, T, V]
        probs = F.softmax(logits, dim=-1)          # [B, T, V]
        B, T, V = probs.shape
        # Flatten for multinomial sampling then reshape
        flat_probs = probs.view(B * T, V)
        sampled_flat = torch.multinomial(flat_probs, num_samples=1).squeeze(-1)  # [B*T]
        sampled = sampled_flat.view(B, T)
        return sampled


# ---------------------------------------------------------------------------
# NCETrainer
# ---------------------------------------------------------------------------

class NCETrainer:
    """Trains a SequenceEnergyFunction with NCE and contrastive divergence."""

    def __init__(
        self,
        model: SequenceEnergyFunction,
        lr: float = 1e-4,
    ) -> None:
        self.model = model
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        # Uniform noise distribution (represented implicitly by random sampling)

    # ------------------------------------------------------------------
    # nce_loss
    # ------------------------------------------------------------------

    def nce_loss(
        self,
        pos_ids: torch.Tensor,
        neg_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Noise contrastive estimation loss.

        Logistic regression: classify real sequences (+1) vs noise (0).

        L = -E[log σ(score(x+))] - E[log σ(-score(x-))]

        where score(x) = -energy(x)  (unnormalized log probability).

        Args:
            pos_ids: [B, T]  real sequences
            neg_ids: [B, T]  noise sequences

        Returns:
            scalar loss
        """
        pos_score = self.model.score(pos_ids)   # [B]
        neg_score = self.model.score(neg_ids)   # [B]

        pos_loss = -F.logsigmoid(pos_score)     # [B]
        neg_loss = -F.logsigmoid(-neg_score)    # [B]

        loss = (pos_loss + neg_loss).mean()
        return loss

    # ------------------------------------------------------------------
    # contrastive_divergence
    # ------------------------------------------------------------------

    def contrastive_divergence(
        self,
        pos_ids: torch.Tensor,
        n_mcmc_steps: int = 5,
    ) -> torch.Tensor:
        """Contrastive Divergence-k (CD-k).

        Starting from corrupted positives, run k Gibbs-like token-flip steps
        to obtain negative samples, then minimize energy(pos) - energy(neg).

        Args:
            pos_ids:      [B, T]
            n_mcmc_steps: number of MCMC steps (k in CD-k)

        Returns:
            scalar loss
        """
        B, T = pos_ids.shape
        vocab_size = self.model.vocab_size

        # Initialise negatives from corrupted positives (no grad needed here)
        neg_ids = NegativeSampler.random_corrupt(pos_ids, corrupt_frac=0.15)

        # Gibbs-like MCMC: at each step randomly pick one position per sequence
        # and propose a random replacement; accept with probability proportional
        # to exp(-energy_new + energy_old)  (Metropolis–Hastings step).
        neg_ids = neg_ids.detach()
        with torch.no_grad():
            for _ in range(n_mcmc_steps):
                # Pick a random position per sequence
                pos_idx = torch.randint(0, T, (B,), device=pos_ids.device)   # [B]
                new_token = torch.randint(0, vocab_size, (B,), device=pos_ids.device)

                # Build proposed sequences
                proposed = neg_ids.clone()
                for b in range(B):
                    proposed[b, pos_idx[b]] = new_token[b]

                # Acceptance probability (MH)
                energy_old = self.model(neg_ids)        # [B]
                energy_new = self.model(proposed)       # [B]
                log_alpha = -(energy_new - energy_old)  # [B]
                accept = torch.rand(B, device=pos_ids.device).log() < log_alpha
                for b in range(B):
                    if accept[b]:
                        neg_ids[b] = proposed[b]

        # CD loss: energy(pos) - energy(neg)
        energy_pos = self.model(pos_ids)
        energy_neg = self.model(neg_ids)
        loss = (energy_pos - energy_neg).mean()
        return loss

    # ------------------------------------------------------------------
    # train_step
    # ------------------------------------------------------------------

    def train_step(
        self,
        input_ids: torch.Tensor,
        negative_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Single optimisation step using NCE loss.

        Args:
            input_ids:    [B, T]  positive (real) sequences
            negative_ids: [B, T]  negative (noise) sequences

        Returns:
            scalar loss (detached)
        """
        self.optimizer.zero_grad()
        loss = self.nce_loss(input_ids, negative_ids)
        loss.backward()
        self.optimizer.step()
        return loss.detach()


# ---------------------------------------------------------------------------
# LangevinSampler
# ---------------------------------------------------------------------------

class LangevinSampler:
    """Langevin dynamics sampler in continuous embedding space."""

    def __init__(
        self,
        energy_fn: SequenceEnergyFunction,
        step_size: float = 0.1,
    ) -> None:
        self.energy_fn = energy_fn
        self.step_size = step_size

    # ------------------------------------------------------------------
    # step
    # ------------------------------------------------------------------

    def step(self, x: torch.Tensor) -> torch.Tensor:
        """One Langevin step: x ← x − α ∇_x E(x) + √(2α) ε

        Because the energy function operates on token ids (integers), we work
        entirely in the continuous embedding space: we pass the embedding
        tensor directly through the encoder (bypassing the discrete embedding
        lookup) and compute gradients w.r.t. the embedding tensor.

        Args:
            x: [B, T, d_model]  continuous embedding tensor

        Returns:
            x_new: [B, T, d_model]
        """
        x = x.detach().requires_grad_(True)

        # Forward pass through encoder layers (skip embedding lookup)
        h = self.energy_fn.pos_enc(x)
        for layer in self.energy_fn.encoder_layers:
            h = layer(h)
        pooled = h.mean(dim=1)
        energy = self.energy_fn.mlp(pooled).squeeze(-1).sum()   # scalar

        # Gradient w.r.t. continuous embedding x
        grad = torch.autograd.grad(energy, x)[0]                # [B, T, d_model]

        noise = torch.randn_like(x) * math.sqrt(2.0 * self.step_size)
        x_new = x.detach() - self.step_size * grad.detach() + noise
        return x_new

    # ------------------------------------------------------------------
    # sample
    # ------------------------------------------------------------------

    def sample(
        self,
        x0: torch.Tensor,
        n_steps: int = 50,
    ) -> torch.Tensor:
        """Run *n_steps* Langevin steps from initial point x0.

        Args:
            x0:      [B, T, d_model]
            n_steps: number of Langevin steps

        Returns:
            x: [B, T, d_model]
        """
        x = x0.clone()
        for _ in range(n_steps):
            x = self.step(x)
        return x


# ---------------------------------------------------------------------------
# EBMReranker
# ---------------------------------------------------------------------------

class EBMReranker:
    """Rerank candidate sequences using the energy function.

    Lower energy  →  better (more likely) sequence.
    """

    def __init__(
        self,
        energy_fn: SequenceEnergyFunction,
        n_candidates: int = 5,
    ) -> None:
        self.energy_fn = energy_fn
        self.n_candidates = n_candidates

    # ------------------------------------------------------------------
    # rerank
    # ------------------------------------------------------------------

    def rerank(self, candidates: List[torch.Tensor]) -> torch.Tensor:
        """Score each candidate and return the one with the lowest energy.

        Args:
            candidates: list of tensors, each [B, T] or [T] (1-D)

        Returns:
            best: [T] or [B, T] — the candidate with the lowest total energy
        """
        with torch.no_grad():
            energies = []
            for cand in candidates:
                if cand.dim() == 1:
                    cand = cand.unsqueeze(0)   # [1, T]
                e = self.energy_fn(cand)       # [B] or [1]
                energies.append(e.mean().item())

        best_idx = int(min(range(len(energies)), key=lambda i: energies[i]))
        return candidates[best_idx]

    # ------------------------------------------------------------------
    # batch_rerank
    # ------------------------------------------------------------------

    def batch_rerank(self, all_candidates: torch.Tensor) -> torch.Tensor:
        """Rerank candidates for every item in a batch.

        Args:
            all_candidates: [B, n_cand, T]

        Returns:
            best: [B, T]
        """
        B, n_cand, T = all_candidates.shape
        with torch.no_grad():
            # Flatten to [B*n_cand, T], score, then reshape to [B, n_cand]
            flat = all_candidates.view(B * n_cand, T)
            energies = self.energy_fn(flat).view(B, n_cand)   # [B, n_cand]

        # Pick argmin along candidate dimension
        best_idx = energies.argmin(dim=1)    # [B]
        # Gather best candidate for each batch item
        best_idx_exp = best_idx.view(B, 1, 1).expand(B, 1, T)
        best = all_candidates.gather(1, best_idx_exp).squeeze(1)  # [B, T]
        return best
