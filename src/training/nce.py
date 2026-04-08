"""
Noise Contrastive Estimation (NCE) and InfoNCE for the Aurelius LLM project.

NCE replaces the expensive full-vocabulary softmax with a binary classification task:
is this token real or noise-sampled? InfoNCE extends this idea to contrastive
representation learning by maximising mutual information between anchor and positive
representations using in-batch negatives.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Noise distribution
# ---------------------------------------------------------------------------

class NoiseDistribution:
    """Smoothed unigram noise distribution for NCE sampling.

    p_noise(w) ∝ freq(w)^alpha   (alpha=0.75 from word2vec)
    """

    def __init__(self, token_freqs: torch.Tensor, alpha: float = 0.75) -> None:
        """
        Args:
            token_freqs: (vocab_size,) token frequency counts.
            alpha: smoothing exponent (0.75 from word2vec).
        """
        smoothed = token_freqs.float().pow(alpha)
        self.probs: torch.Tensor = smoothed / smoothed.sum()

    # ------------------------------------------------------------------
    def sample(self, n: int, device=None) -> torch.Tensor:
        """Sample *n* noise token indices.

        Returns:
            (n,) int64 tensor of sampled token ids.
        """
        probs = self.probs
        if device is not None:
            probs = probs.to(device)
        return torch.multinomial(probs, num_samples=n, replacement=True)

    # ------------------------------------------------------------------
    def log_prob(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Log probability of *token_ids* under the noise distribution.

        Args:
            token_ids: arbitrary-shape int64 tensor of token indices.

        Returns:
            Float tensor of the same shape as *token_ids*.
        """
        probs = self.probs.to(token_ids.device)
        return torch.log(probs[token_ids])


# ---------------------------------------------------------------------------
# NCE loss
# ---------------------------------------------------------------------------

class NCELoss(nn.Module):
    """Noise Contrastive Estimation loss.

    For each real token we sample *k* noise tokens and train a binary
    classifier:

        P(real | token, context) = sigmoid(score - log(k * q(w)))

    where score = hidden @ output_emb[w] + bias[w] and q(w) is the noise
    probability.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        k: int = 20,
        noise_dist: NoiseDistribution | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.k = k
        self.noise_dist = noise_dist

        # Output (target) embeddings — can be weight-tied with input embeddings
        self.output_embeddings = nn.Embedding(vocab_size, d_model)
        self.output_bias = nn.Parameter(torch.zeros(vocab_size))

    # ------------------------------------------------------------------
    def _score(
        self,
        hidden: torch.Tensor,   # (..., d_model)
        token_ids: torch.Tensor,  # (...,)
    ) -> torch.Tensor:
        """Dot-product score + bias for each (hidden, token_id) pair."""
        emb = self.output_embeddings(token_ids)   # (..., d_model)
        bias = self.output_bias[token_ids]         # (...,)
        return (hidden * emb).sum(-1) + bias       # (...,)

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden: torch.Tensor,     # (B, S, d_model)
        targets: torch.Tensor,    # (B, S) true next tokens
        noise_dist: NoiseDistribution | None = None,
    ) -> torch.Tensor:
        """Compute the NCE loss.

        Steps per position:
        1. Score the true token.
        2. Sample k noise tokens from the noise distribution.
        3. Score each noise token.
        4. Compute the binary-classification objective.

        Returns:
            Scalar loss tensor.
        """
        dist = noise_dist if noise_dist is not None else self.noise_dist

        B, S, D = hidden.shape
        device = hidden.device

        # ---- true token scores ------------------------------------------
        # targets: (B, S) → score: (B, S)
        s_true = self._score(hidden, targets)

        if dist is not None:
            log_q_true = dist.log_prob(targets)  # (B, S)
        else:
            log_q_true = torch.full_like(s_true, -torch.log(torch.tensor(float(self.vocab_size))))

        # Correction: subtract log(k * q(target))
        delta_true = s_true - (torch.log(torch.tensor(float(self.k), device=device)) + log_q_true)
        loss_true = -F.logsigmoid(delta_true)  # (B, S)

        # ---- noise token scores -----------------------------------------
        # Sample k noise tokens (shared across all positions for efficiency)
        if dist is not None:
            noise_ids = dist.sample(self.k, device=device)  # (k,)
        else:
            noise_ids = torch.randint(0, self.vocab_size, (self.k,), device=device)

        # Expand hidden for noise scoring: (B, S, k, D)
        hidden_exp = hidden.unsqueeze(2).expand(B, S, self.k, D)

        # noise_ids_exp: (B, S, k)
        noise_ids_exp = noise_ids.view(1, 1, self.k).expand(B, S, self.k)

        # Embeddings: (k, D) → (B, S, k, D)
        noise_emb = self.output_embeddings(noise_ids)  # (k, D)
        noise_emb_exp = noise_emb.view(1, 1, self.k, D).expand(B, S, self.k, D)

        # Bias: (k,) → (B, S, k)
        noise_bias = self.output_bias[noise_ids].view(1, 1, self.k).expand(B, S, self.k)

        s_noise = (hidden_exp * noise_emb_exp).sum(-1) + noise_bias  # (B, S, k)

        if dist is not None:
            log_q_noise = dist.log_prob(noise_ids_exp)  # (B, S, k)
        else:
            log_q_noise = torch.full_like(s_noise, -torch.log(torch.tensor(float(self.vocab_size))))

        delta_noise = s_noise - (torch.log(torch.tensor(float(self.k), device=device)) + log_q_noise)
        loss_noise = -F.logsigmoid(-delta_noise)  # (B, S, k)

        # ---- combine ----------------------------------------------------
        loss = loss_true + loss_noise.sum(-1)   # (B, S)
        return loss.mean()


# ---------------------------------------------------------------------------
# InfoNCE loss
# ---------------------------------------------------------------------------

class InfoNCELoss(nn.Module):
    """InfoNCE contrastive loss with in-batch negatives.

    Given (anchor, positive) pairs, all other positives in the batch serve as
    negatives:

        L = -log[ exp(sim(z_a, z_p)/tau) / Σ_j exp(sim(z_a, z_j)/tau) ]

    where z are L2-normalised projections.
    """

    def __init__(
        self,
        d_model: int,
        projection_dim: int = 128,
        temperature: float = 0.07,
    ) -> None:
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, projection_dim),
        )
        self.temperature = temperature

    # ------------------------------------------------------------------
    def forward(
        self,
        anchors: torch.Tensor,    # (B, d_model)
        positives: torch.Tensor,  # (B, d_model)
    ) -> torch.Tensor:
        """Compute InfoNCE with in-batch negatives.

        Returns:
            Scalar loss tensor.
        """
        B = anchors.shape[0]

        z_a = F.normalize(self.projector(anchors), dim=-1)    # (B, proj_dim)
        z_p = F.normalize(self.projector(positives), dim=-1)  # (B, proj_dim)

        # Similarity matrix: (B, B)
        logits = torch.matmul(z_a, z_p.T) / self.temperature

        # Labels: diagonal (each anchor matches its own positive)
        labels = torch.arange(B, device=anchors.device)

        return F.cross_entropy(logits, labels)


# ---------------------------------------------------------------------------
# Self-normalised NCE
# ---------------------------------------------------------------------------

class SelfNormalizedNCE(nn.Module):
    """Self-normalised NCE — encourages log Z ≈ 0.

    Adds a regularisation term that penalises the log partition function
    deviating from zero, eliminating the need to estimate Z explicitly.

        total_loss = NCE_loss + lambda * (log Z)^2

    where Z is estimated from the sampled noise scores.
    """

    def __init__(
        self,
        nce_loss: NCELoss,
        normalization_lambda: float = 1.0,
    ) -> None:
        super().__init__()
        self.nce_loss = nce_loss
        self.lam = normalization_lambda

    # ------------------------------------------------------------------
    def forward(
        self,
        hidden: torch.Tensor,    # (B, S, d_model)
        targets: torch.Tensor,   # (B, S)
        noise_dist: NoiseDistribution | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute total loss and normalisation penalty.

        Returns:
            (total_loss, normalization_penalty) — both scalar tensors.
        """
        B, S, D = hidden.shape
        device = hidden.device
        dist = noise_dist if noise_dist is not None else self.nce_loss.noise_dist

        # NCE loss (standard)
        base_loss = self.nce_loss(hidden, targets, noise_dist=dist)

        # --- estimate log Z from a sample of vocab scores ----------------
        # Sample a small set of tokens to estimate Z
        k_norm = max(self.nce_loss.k, 50)
        if dist is not None:
            sample_ids = dist.sample(k_norm, device=device)
        else:
            sample_ids = torch.randint(
                0, self.nce_loss.vocab_size, (k_norm,), device=device
            )

        # Compute scores for these tokens against mean hidden state
        h_mean = hidden.detach().mean(dim=(0, 1), keepdim=False)  # (D,)
        emb = self.nce_loss.output_embeddings(sample_ids)         # (k_norm, D)
        bias = self.nce_loss.output_bias[sample_ids]              # (k_norm,)
        scores = (emb * h_mean.unsqueeze(0)).sum(-1) + bias       # (k_norm,)

        # log Z ≈ logsumexp over sample (biased but differentiable)
        log_z = torch.logsumexp(scores, dim=0)

        # Penalty: (log Z)^2  → encourages log Z → 0
        penalty = self.lam * log_z.pow(2)

        total = base_loss + penalty
        return total, penalty
