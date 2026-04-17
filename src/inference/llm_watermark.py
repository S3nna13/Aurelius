"""
LLM Watermarking — Kirchenbauer et al. 2023 ("green/red list" watermark)
plus a learned watermark detector.

Pure PyTorch only. No external dependencies beyond stdlib + torch.
"""

from __future__ import annotations

import math
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# GreenRedPartitioner
# ---------------------------------------------------------------------------

class GreenRedPartitioner:
    """Partition vocabulary into green/red lists per context token."""

    def __init__(self, vocab_size: int, gamma: float = 0.5, seed_key: int = 42) -> None:
        self.vocab_size = vocab_size
        self.gamma = gamma
        self.seed_key = seed_key
        self._n_green = int(math.floor(gamma * vocab_size))

    # ------------------------------------------------------------------
    def partition(self, context_hash: int) -> Tuple[Tensor, Tensor]:
        """Return (green_ids, red_ids) for the given context hash.

        Uses (context_hash XOR seed_key) as the RNG seed so that
        different contexts yield different partitions deterministically.
        """
        seed = (context_hash ^ self.seed_key) & 0xFFFF_FFFF_FFFF_FFFF
        # torch.Generator for reproducible, independent partitions
        g = torch.Generator()
        g.manual_seed(seed)
        perm = torch.randperm(self.vocab_size, generator=g)
        green_ids = perm[: self._n_green]
        red_ids = perm[self._n_green :]
        return green_ids, red_ids

    # ------------------------------------------------------------------
    def context_hash(self, token_ids: Tensor) -> int:
        """Hash of the last h=1 tokens (single previous-token hash)."""
        return int(token_ids[-1:].sum().item())

    # ------------------------------------------------------------------
    def green_fraction(self) -> float:
        """Return gamma."""
        return self.gamma


# ---------------------------------------------------------------------------
# WatermarkLogitsProcessor
# ---------------------------------------------------------------------------

class WatermarkLogitsProcessor:
    """Modify logits to bias toward green-list tokens."""

    def __init__(self, partitioner: GreenRedPartitioner, delta: float = 2.0) -> None:
        self.partitioner = partitioner
        self.delta = delta

    # ------------------------------------------------------------------
    def process(self, input_ids: Tensor, logits: Tensor) -> Tensor:
        """Return watermarked logits of shape (B, V).

        For each batch item b, compute context_hash from input_ids[b, -1],
        retrieve the green list, and add delta to those positions.
        """
        # input_ids: (B, T)  logits: (B, V)
        watermarked = logits.clone()
        batch_size = logits.shape[0]
        for b in range(batch_size):
            ctx_hash = self.partitioner.context_hash(input_ids[b])
            green_ids, _ = self.partitioner.partition(ctx_hash)
            watermarked[b, green_ids] = watermarked[b, green_ids] + self.delta
        return watermarked


# ---------------------------------------------------------------------------
# WatermarkDetector
# ---------------------------------------------------------------------------

class WatermarkDetector:
    """Statistical test for watermark presence (z-score based)."""

    def __init__(self, partitioner: GreenRedPartitioner, z_threshold: float = 4.0) -> None:
        self.partitioner = partitioner
        self.z_threshold = z_threshold

    # ------------------------------------------------------------------
    def count_green_tokens(self, token_ids: Tensor) -> Tuple[int, int]:
        """Count how many tokens fall in their respective green lists.

        For position i, the context is token_ids[i-1].  Position 0 is
        skipped because there is no preceding token.
        """
        n_total = 0
        n_green = 0
        seq_len = token_ids.shape[0]
        for i in range(1, seq_len):
            ctx_hash = self.partitioner.context_hash(token_ids[i - 1 : i])
            green_ids, _ = self.partitioner.partition(ctx_hash)
            token = int(token_ids[i].item())
            if token in set(green_ids.tolist()):
                n_green += 1
            n_total += 1
        return n_green, n_total

    # ------------------------------------------------------------------
    def z_score(self, n_green: int, n_total: int) -> float:
        """Compute z-score for the green-token count."""
        gamma = self.partitioner.green_fraction()
        if n_total == 0:
            return 0.0
        numerator = n_green - gamma * n_total
        denominator = math.sqrt(n_total * gamma * (1.0 - gamma))
        if denominator == 0.0:
            return 0.0
        return float(numerator / denominator)

    # ------------------------------------------------------------------
    def detect(self, token_ids: Tensor) -> Tuple[bool, float, float]:
        """Detect watermark presence.

        Returns:
            is_watermarked: True if z_score > z_threshold
            z:              computed z-score
            p_value:        one-tailed p-value  1 - Φ(z)
        """
        n_green, n_total = self.count_green_tokens(token_ids)
        z = self.z_score(n_green, n_total)
        # p_value = 1 - Φ(z),  Φ(z) ≈ 0.5*(1 + erf(z/√2))
        phi_z = 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        p_value = max(0.0, min(1.0, 1.0 - phi_z))
        is_watermarked = z > self.z_threshold
        return is_watermarked, float(z), float(p_value)


# ---------------------------------------------------------------------------
# LearnedWatermarkDetector
# ---------------------------------------------------------------------------

class LearnedWatermarkDetector(nn.Module):
    """Trainable binary classifier to detect watermarked sequences."""

    def __init__(self, vocab_size: int, embed_dim: int = 32, n_layers: int = 2) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        mlp_layers: List[nn.Module] = []
        for _ in range(n_layers):
            mlp_layers.append(nn.Linear(embed_dim, embed_dim))
            mlp_layers.append(nn.ReLU())
        self.mlp = nn.Sequential(*mlp_layers)
        self.head = nn.Linear(embed_dim, 1)

    # ------------------------------------------------------------------
    def forward(self, token_ids: Tensor) -> Tensor:
        """Return watermark probability, shape (B,) in (0, 1).

        token_ids: (B, T)
        """
        # Embed + mean-pool over sequence dimension
        x = self.embedding(token_ids)          # (B, T, D)
        x = x.mean(dim=1)                      # (B, D)
        x = self.mlp(x)                        # (B, D)
        x = self.head(x).squeeze(-1)           # (B,)
        return torch.sigmoid(x)

    # ------------------------------------------------------------------
    def loss(self, token_ids: Tensor, labels: Tensor) -> Tensor:
        """Binary cross-entropy loss.

        token_ids: (B, T)
        labels:    (B,) float in {0, 1}
        """
        probs = self.forward(token_ids)        # (B,)
        return F.binary_cross_entropy(probs, labels)


# ---------------------------------------------------------------------------
# WatermarkBenchmark
# ---------------------------------------------------------------------------

class WatermarkBenchmark:
    """Evaluate watermark quality via TPR/FPR, KL distortion, and perplexity."""

    def __init__(self, detector: WatermarkDetector) -> None:
        self.detector = detector

    # ------------------------------------------------------------------
    def tpr_at_fpr(
        self,
        watermarked_sequences: List[Tensor],
        clean_sequences: List[Tensor],
        fpr_target: float = 0.01,
    ) -> float:
        """True-positive rate on watermarked sequences at the given FPR.

        FPR is controlled by finding the z-score threshold on clean sequences.
        """
        clean_z: List[float] = []
        for seq in clean_sequences:
            _, z, _ = self.detector.detect(seq)
            clean_z.append(z)

        wm_z: List[float] = []
        for seq in watermarked_sequences:
            _, z, _ = self.detector.detect(seq)
            wm_z.append(z)

        # Find threshold on clean sequences for the target FPR
        clean_z_sorted = sorted(clean_z, reverse=True)
        n_clean = len(clean_z_sorted)
        if n_clean == 0:
            return 0.0

        # Number of false positives allowed
        n_fp = max(1, int(math.ceil(fpr_target * n_clean)))
        if n_fp <= len(clean_z_sorted):
            threshold = clean_z_sorted[n_fp - 1]
        else:
            threshold = float("-inf")

        # Measure TPR on watermarked sequences
        n_wm = len(wm_z)
        if n_wm == 0:
            return 0.0
        tp = sum(1 for z in wm_z if z >= threshold)
        return float(tp) / float(n_wm)

    # ------------------------------------------------------------------
    def distortion_score(
        self, original_logits: Tensor, watermarked_logits: Tensor
    ) -> float:
        """Mean KL divergence KL(original || watermarked) over batch.

        logits shape: (B, V) or (V,)
        """
        if original_logits.dim() == 1:
            original_logits = original_logits.unsqueeze(0)
            watermarked_logits = watermarked_logits.unsqueeze(0)

        log_p = F.log_softmax(original_logits, dim=-1)
        log_q = F.log_softmax(watermarked_logits, dim=-1)
        p = torch.exp(log_p)

        # KL(p || q) = sum_x p(x) * (log p(x) - log q(x))
        kl = (p * (log_p - log_q)).sum(dim=-1)   # (B,)
        return float(kl.mean().item())

    # ------------------------------------------------------------------
    def perplexity_impact(
        self, watermarked_logprobs: Tensor, clean_logprobs: Tensor
    ) -> float:
        """Ratio of watermarked perplexity to clean perplexity.

        logprobs: (T,) token log-probabilities (negative values expected).
        perplexity = exp(-mean(logprobs))
        """
        wm_ppl = math.exp(-float(watermarked_logprobs.mean().item()))
        cl_ppl = math.exp(-float(clean_logprobs.mean().item()))
        if cl_ppl == 0.0:
            return float("inf")
        return float(wm_ppl / cl_ppl)
