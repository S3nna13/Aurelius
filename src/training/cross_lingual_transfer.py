"""Cross-lingual transfer learning: language-agnostic representations,
alignment losses, and zero-shot cross-lingual transfer training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Gradient Reversal Layer
# ---------------------------------------------------------------------------

class _GradReversalFn(torch.autograd.Function):
    """Identity in forward; multiply gradient by -lambda in backward."""

    @staticmethod
    def forward(ctx, x: Tensor, lam: float) -> Tensor:  # type: ignore[override]
        ctx.lam = lam
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        return -ctx.lam * grad_output, None


# ---------------------------------------------------------------------------
# LanguageIdentifier
# ---------------------------------------------------------------------------

class LanguageIdentifier(nn.Module):
    """Classifier head that predicts language from a pooled representation."""

    def __init__(self, d_model: int, n_languages: int) -> None:
        super().__init__()
        self.classifier = nn.Linear(d_model, n_languages)

    def forward(self, pooled: Tensor) -> Tensor:
        """
        Args:
            pooled: [B, d_model]
        Returns:
            lang_logits: [B, n_languages]
        """
        return self.classifier(pooled)

    def predict(self, pooled: Tensor) -> Tensor:
        """
        Args:
            pooled: [B, d_model]
        Returns:
            lang_ids: [B]  (argmax over n_languages dimension)
        """
        with torch.no_grad():
            logits = self.forward(pooled)
        return logits.argmax(dim=-1)


# ---------------------------------------------------------------------------
# LanguageAdversary
# ---------------------------------------------------------------------------

class LanguageAdversary(nn.Module):
    """Adversarial language classifier using a Gradient Reversal Layer.

    Forces encoder representations to be language-agnostic by reversing
    gradients flowing back through the GRL.
    """

    def __init__(
        self,
        d_model: int,
        n_languages: int,
        lambda_adv: float = 0.1,
    ) -> None:
        super().__init__()
        self.lambda_adv = lambda_adv
        self.identifier = LanguageIdentifier(d_model, n_languages)

    def forward(self, features: Tensor) -> Tensor:
        """
        Args:
            features: [B, d_model]
        Returns:
            lang_logits: [B, n_languages]
        """
        reversed_features = _GradReversalFn.apply(features, self.lambda_adv)
        return self.identifier(reversed_features)


# ---------------------------------------------------------------------------
# AlignmentLoss
# ---------------------------------------------------------------------------

class AlignmentLoss:
    """Collection of alignment loss functions for cross-lingual training."""

    @staticmethod
    def mean_alignment(src_emb: Tensor, tgt_emb: Tensor) -> Tensor:
        """Mean cosine distance between parallel sentence embeddings.

        L = 1 - cosine_sim(mean(src_emb, T), mean(tgt_emb, T))

        Args:
            src_emb: [B, T, d]
            tgt_emb: [B, T, d]
        Returns:
            scalar loss in [0, 2]
        """
        src_pooled = src_emb.mean(dim=1)  # [B, d]
        tgt_pooled = tgt_emb.mean(dim=1)  # [B, d]
        cos_sim = F.cosine_similarity(src_pooled, tgt_pooled, dim=-1)  # [B]
        return (1.0 - cos_sim).mean()

    @staticmethod
    def contrastive_alignment(
        src_emb: Tensor,
        tgt_emb: Tensor,
        temperature: float = 0.07,
    ) -> Tensor:
        """InfoNCE contrastive loss between parallel sentence embeddings.

        L = -log(exp(sim(si, ti)/tau) / sum_j exp(sim(si, tj)/tau))

        Args:
            src_emb: [B, d]
            tgt_emb: [B, d]
            temperature: scaling factor tau
        Returns:
            scalar loss (non-negative)
        """
        src_norm = F.normalize(src_emb, dim=-1)  # [B, d]
        tgt_norm = F.normalize(tgt_emb, dim=-1)  # [B, d]

        # Similarity matrix [B, B]
        logits = torch.matmul(src_norm, tgt_norm.T) / temperature

        B = src_emb.size(0)
        labels = torch.arange(B, device=src_emb.device)

        loss_s2t = F.cross_entropy(logits, labels)
        loss_t2s = F.cross_entropy(logits.T, labels)
        return (loss_s2t + loss_t2s) / 2.0

    @staticmethod
    def word_alignment(
        src: Tensor,
        tgt: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """Soft word alignment via dot-product attention with entropy regularization.

        Args:
            src: [B, T_s, d]
            tgt: [B, T_t, d]
        Returns:
            attn:  [B, T_s, T_t]  -- soft alignment weights
            loss:  scalar          -- entropy regularization (encourage confident alignment)
        """
        d = src.size(-1)
        scores = torch.bmm(src, tgt.transpose(1, 2)) / math.sqrt(d)
        attn = F.softmax(scores, dim=-1)  # [B, T_s, T_t]

        eps = 1e-8
        entropy = -(attn * (attn + eps).log()).sum(dim=-1)  # [B, T_s]
        loss = entropy.mean()

        return attn, loss


# ---------------------------------------------------------------------------
# CrossLingualEncoder
# ---------------------------------------------------------------------------

class _TransformerEncoderBlock(nn.Module):
    """Single transformer encoder layer (self-attention + FFN)."""

    def __init__(self, d_model: int, n_heads: int = 4, ffn_mult: int = 4) -> None:
        super().__init__()
        # Ensure d_model divisible by n_heads
        while n_heads > 1 and d_model % n_heads != 0:
            n_heads -= 1
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * ffn_mult),
            nn.GELU(),
            nn.Linear(d_model * ffn_mult, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class CrossLingualEncoder(nn.Module):
    """Shared transformer encoder for all languages with adversarial training."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_layers: int = 2,
        n_languages: int = 5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.language_embedding = nn.Embedding(n_languages, d_model)
        self.layers = nn.ModuleList(
            [_TransformerEncoderBlock(d_model) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)
        self.adversary = LanguageAdversary(d_model, n_languages)

    def forward(
        self,
        input_ids: Tensor,
        lang_ids: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """
        Args:
            input_ids: [B, T]  token indices
            lang_ids:  [B]     language indices
        Returns:
            token_repr: [B, T, d_model]
            pooled:     [B, d_model]   (mean-pooled)
        """
        tok_emb = self.token_embedding(input_ids)           # [B, T, d]
        lang_emb = self.language_embedding(lang_ids)        # [B, d]
        x = tok_emb + lang_emb.unsqueeze(1)                 # broadcast over T

        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)

        pooled = x.mean(dim=1)  # [B, d]
        return x, pooled


# ---------------------------------------------------------------------------
# ZeroShotTransferTrainer
# ---------------------------------------------------------------------------

class ZeroShotTransferTrainer:
    """Trains cross-lingual transfer with task, alignment, and adversarial losses."""

    def __init__(
        self,
        encoder: CrossLingualEncoder,
        task_head: nn.Module,
        lr: float = 1e-4,
        alpha_align: float = 0.5,
        beta_adv: float = 0.1,
        temperature: float = 0.07,
    ) -> None:
        self.encoder = encoder
        self.task_head = task_head
        self.alignment_loss = AlignmentLoss()
        self.alpha_align = alpha_align
        self.beta_adv = beta_adv
        self.temperature = temperature

        params = list(encoder.parameters()) + list(task_head.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)

    def train_step(
        self,
        src_ids: Tensor,
        src_lang: Tensor,
        tgt_ids: Tensor,
        tgt_lang: Tensor,
        labels: Tensor,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Single training step.

        total = task_loss(src) + alpha * align_loss(src_emb, tgt_emb) + beta * adv_loss

        Args:
            src_ids:  [B, T]  source language token ids
            src_lang: [B]     source language ids
            tgt_ids:  [B, T]  target language token ids
            tgt_lang: [B]     target language ids
            labels:   [B]     task labels (class indices)
        Returns:
            total_loss: scalar Tensor
            loss_dict:  dict with keys 'task', 'align', 'adv', 'total'
        """
        self.encoder.train()
        self.task_head.train()
        self.optimizer.zero_grad()

        # Encode source and target
        _src_repr, src_pooled = self.encoder(src_ids, src_lang)
        _tgt_repr, tgt_pooled = self.encoder(tgt_ids, tgt_lang)

        # Task loss on source
        task_logits = self.task_head(src_pooled)
        task_loss = F.cross_entropy(task_logits, labels)

        # Contrastive alignment loss
        align_loss = AlignmentLoss.contrastive_alignment(
            src_pooled, tgt_pooled, temperature=self.temperature
        )

        # Adversarial loss (GRL inside adversary reverses gradient into encoder)
        adv_logits = self.encoder.adversary(src_pooled)
        adv_loss = F.cross_entropy(adv_logits, src_lang)

        total_loss = (
            task_loss
            + self.alpha_align * align_loss
            + self.beta_adv * adv_loss
        )
        total_loss.backward()
        self.optimizer.step()

        loss_dict: Dict[str, float] = {
            "task": task_loss.item(),
            "align": align_loss.item(),
            "adv": adv_loss.item(),
            "total": total_loss.item(),
        }
        return total_loss, loss_dict

    @torch.no_grad()
    def evaluate_transfer(
        self,
        src_ids: Tensor,
        src_lang: Tensor,
        tgt_ids: Tensor,
        tgt_lang: Tensor,
        labels: Tensor,
    ) -> float:
        """Evaluate zero-shot accuracy on target language.

        Encodes target-language inputs through the shared encoder (no target
        language fine-tuning) and measures classification accuracy.

        Returns:
            accuracy in [0, 1]
        """
        self.encoder.eval_mode = True  # informational only
        self.encoder.eval()
        self.task_head.eval()

        _tgt_repr, tgt_pooled = self.encoder(tgt_ids, tgt_lang)
        task_logits = self.task_head(tgt_pooled)
        preds = task_logits.argmax(dim=-1)
        accuracy = (preds == labels).float().mean().item()
        return float(accuracy)


# ---------------------------------------------------------------------------
# CrossLingualConfig
# ---------------------------------------------------------------------------

@dataclass
class CrossLingualConfig:
    """Hyperparameter configuration for cross-lingual transfer training."""

    d_model: int = 32
    vocab_size: int = 64
    n_layers: int = 2
    n_languages: int = 3

    lambda_adv: float = 0.1
    alpha_align: float = 0.5
    beta_adv: float = 0.1

    lr: float = 1e-4
    temperature: float = 0.07
