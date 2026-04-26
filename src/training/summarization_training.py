"""
Abstractive Summarization Training
===================================
Encoder-decoder summarization with coverage mechanisms,
extractive-abstractive dual training, and summarization-specific losses.

Pure PyTorch only — no external ML libraries.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# CoverageVector
# ---------------------------------------------------------------------------


class CoverageVector:
    """Tracks accumulated attention weights to penalise re-attending."""

    def __init__(self) -> None:
        self.coverage: Tensor | None = None  # [B, T_src]

    def reset(self, batch_size: int, src_len: int) -> None:
        """Initialise / clear coverage to zeros."""
        self.coverage = torch.zeros(batch_size, src_len)

    def update(self, attn_weights: Tensor) -> None:
        """Accumulate attention weights into coverage.

        Args:
            attn_weights: [B, T_src] attention distribution for the current
                          decoder step.
        """
        if self.coverage is None:
            self.coverage = attn_weights.detach().clone()
        else:
            self.coverage = self.coverage + attn_weights.detach()

    def coverage_loss(self, attn_weights: Tensor) -> Tensor:
        """Compute coverage loss for one decoder step.

        L_cov = sum_i min(a_it, c_it)  (penalise re-attending).

        Args:
            attn_weights: [B, T_src] attention for the current step.

        Returns:
            Scalar loss.
        """
        if self.coverage is None:
            # No accumulated coverage yet — no penalty.
            return torch.tensor(0.0, dtype=attn_weights.dtype, device=attn_weights.device)
        cov = self.coverage.to(attn_weights.device)
        loss = torch.min(attn_weights, cov).sum()
        return loss


# ---------------------------------------------------------------------------
# ExtractivePseudoLabel
# ---------------------------------------------------------------------------


class ExtractivePseudoLabel:
    """Utilities for computing extractive oracle labels via ROUGE-1 overlap."""

    @staticmethod
    def _token_overlap(a: list[int], b: list[int]) -> float:
        """Unigram F1 (ROUGE-1) between two token lists."""
        if not a or not b:
            return 0.0
        set_a = set(a)
        set_b = set(b)
        overlap = len(set_a & set_b)
        precision = overlap / len(set_a)
        recall = overlap / len(set_b)
        if precision + recall == 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    def compute_rouge_scores(
        self,
        src_tokens: list[list[int]],
        tgt_tokens: list[int],
    ) -> list[float]:
        """ROUGE-1 F1 score for each source sentence against the target.

        Args:
            src_tokens: List of sentences; each sentence is a list of token ids.
            tgt_tokens: Reference summary as a flat list of token ids.

        Returns:
            List of float scores in [0, 1], one per source sentence.
        """
        scores = [self._token_overlap(sent, tgt_tokens) for sent in src_tokens]
        return scores

    def extract_oracle(
        self,
        src_sentences: list[list[int]],
        tgt_tokens: list[int],
        n_sent: int = 3,
    ) -> tuple[list[int], list[float]]:
        """Greedy oracle extraction: iteratively pick sentences with maximum
        marginal ROUGE-1 gain over the already-selected set.

        Args:
            src_sentences: List of source sentences (token id lists).
            tgt_tokens:    Target summary tokens.
            n_sent:        Number of sentences to select.

        Returns:
            (selected_indices, marginal_scores) — both of length min(n_sent, N).
        """
        n_sent = min(n_sent, len(src_sentences))
        selected: list[int] = []
        selected_scores: list[float] = []
        accumulated: list[int] = []

        remaining = list(range(len(src_sentences)))

        for _ in range(n_sent):
            if not remaining:
                break
            best_idx = -1
            best_gain = -1.0
            current_score = self._token_overlap(accumulated, tgt_tokens)

            for idx in remaining:
                candidate = accumulated + src_sentences[idx]
                score = self._token_overlap(candidate, tgt_tokens)
                gain = score - current_score
                if gain > best_gain:
                    best_gain = gain
                    best_idx = idx

            if best_idx == -1:
                best_idx = remaining[0]
                best_gain = 0.0

            selected.append(best_idx)
            selected_scores.append(best_gain)
            accumulated = accumulated + src_sentences[best_idx]
            remaining.remove(best_idx)

        return selected, selected_scores


# ---------------------------------------------------------------------------
# SummarizationLoss
# ---------------------------------------------------------------------------


class SummarizationLoss:
    """Collection of loss functions for summarization training."""

    def __init__(
        self,
        coverage_lambda: float = 1.0,
        length_penalty: float = 0.6,
        pad_id: int = 0,
    ) -> None:
        self.coverage_lambda = coverage_lambda
        self.length_penalty = length_penalty
        self.pad_id = pad_id

    def seq2seq_loss(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Cross-entropy loss ignoring pad tokens.

        Args:
            logits:  [B, T_tgt, V]
            targets: [B, T_tgt] integer token ids

        Returns:
            Scalar loss.
        """
        B, T, V = logits.shape
        loss = F.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            ignore_index=self.pad_id,
        )
        return loss

    def coverage_augmented_loss(
        self,
        logits: Tensor,
        targets: Tensor,
        coverage_loss: Tensor,
    ) -> Tensor:
        """seq2seq CE loss augmented with coverage penalty.

        Args:
            logits:        [B, T_tgt, V]
            targets:       [B, T_tgt]
            coverage_loss: Scalar pre-computed coverage loss.

        Returns:
            Scalar: seq2seq_loss + coverage_lambda * coverage_loss
        """
        base = self.seq2seq_loss(logits, targets)
        return base + self.coverage_lambda * coverage_loss

    def length_penalty_loss(
        self,
        logits: Tensor,
        targets: Tensor,
        gen_len: int,
    ) -> Tensor:
        """Penalise generated sequences much shorter/longer than reference.

        Uses the GNMT-style length normalisation factor and measures deviation
        from the reference length.

        Args:
            logits:  [B, T_tgt, V]
            targets: [B, T_tgt]
            gen_len: Length of the generated sequence.

        Returns:
            Scalar loss: base CE loss scaled by length deviation penalty.
        """
        ref_len = targets.shape[1]
        # GNMT length penalty: ((5 + len) / 6)^alpha
        lp_gen = ((5.0 + gen_len) / 6.0) ** self.length_penalty
        lp_ref = ((5.0 + ref_len) / 6.0) ** self.length_penalty
        # Penalise the ratio deviation from 1
        ratio_penalty = abs(lp_gen / (lp_ref + 1e-8) - 1.0)
        base = self.seq2seq_loss(logits, targets)
        return base * (1.0 + ratio_penalty)


# ---------------------------------------------------------------------------
# ExtractiveAbstractiveTrainer
# ---------------------------------------------------------------------------


class ExtractiveAbstractiveTrainer:
    """Joint extractive + abstractive training loop."""

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        lr: float = 1e-4,
    ) -> None:
        self.encoder = encoder
        self.decoder = decoder
        params = list(encoder.parameters()) + list(decoder.parameters())
        self.optimizer = torch.optim.Adam(params, lr=lr)
        self._loss_fn = SummarizationLoss()

    def extractive_step(
        self,
        src_ids: Tensor,
        extract_labels: Tensor,
    ) -> Tensor:
        """Binary classification: which source positions to include.

        The encoder produces per-token representations; a linear probe maps
        each to a binary logit.

        Args:
            src_ids:        [B, T_src] integer token ids (used as float input
                            directly for simplicity in the training harness).
            extract_labels: [B, T_src] binary labels (0/1).

        Returns:
            Scalar BCE loss.
        """
        self.optimizer.zero_grad()
        enc_out = self.encoder(src_ids)  # [B, T_src, d_model]
        # The encoder must expose .extract_head (a linear -> 1) or we fall back
        # to the last dim as a logit.
        if hasattr(self.encoder, "extract_head"):
            logits = self.encoder.extract_head(enc_out).squeeze(-1)  # [B, T_src]
        else:
            logits = enc_out[..., 0]  # [B, T_src] fallback

        loss = F.binary_cross_entropy_with_logits(
            logits,
            extract_labels.float(),
        )
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def abstractive_step(
        self,
        src_ids: Tensor,
        tgt_ids: Tensor,
    ) -> Tensor:
        """Standard seq2seq cross-entropy step.

        Args:
            src_ids: [B, T_src]
            tgt_ids: [B, T_tgt]

        Returns:
            Scalar CE loss.
        """
        self.optimizer.zero_grad()
        enc_out = self.encoder(src_ids)  # [B, T_src, d_model]
        logits = self.decoder(tgt_ids, enc_out)  # [B, T_tgt, V]
        loss = self._loss_fn.seq2seq_loss(logits, tgt_ids)
        loss.backward()
        self.optimizer.step()
        return loss.detach()

    def joint_step(
        self,
        src_ids: Tensor,
        tgt_ids: Tensor,
        extract_labels: Tensor,
        alpha: float = 0.5,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Combined extractive + abstractive step.

        Args:
            src_ids:        [B, T_src]
            tgt_ids:        [B, T_tgt]
            extract_labels: [B, T_src] binary
            alpha:          Weight for extractive loss.

        Returns:
            (total_loss, extractive_loss, abstractive_loss) — all detached scalars.
        """
        self.optimizer.zero_grad()

        enc_out = self.encoder(src_ids)  # [B, T_src, d_model]

        # Extractive head
        if hasattr(self.encoder, "extract_head"):
            ext_logits = self.encoder.extract_head(enc_out).squeeze(-1)
        else:
            ext_logits = enc_out[..., 0]

        ext_loss = F.binary_cross_entropy_with_logits(
            ext_logits,
            extract_labels.float(),
        )

        # Abstractive head
        logits = self.decoder(tgt_ids, enc_out)  # [B, T_tgt, V]
        abs_loss = self._loss_fn.seq2seq_loss(logits, tgt_ids)

        total = alpha * ext_loss + (1.0 - alpha) * abs_loss
        total.backward()
        self.optimizer.step()

        return total.detach(), ext_loss.detach(), abs_loss.detach()


# ---------------------------------------------------------------------------
# LeadBiasAugmentation
# ---------------------------------------------------------------------------


class LeadBiasAugmentation:
    """Boost the importance of leading tokens (lead bias heuristic)."""

    def __init__(self, lead_frac: float = 0.3) -> None:
        if not 0.0 < lead_frac <= 1.0:
            raise ValueError(f"lead_frac must be in (0, 1], got {lead_frac}")
        self.lead_frac = lead_frac

    def compute_position_bias(self, seq_len: int) -> Tensor:
        """Exponential decay position weight: w_i = 1 / (1 + i).

        Args:
            seq_len: Length of the sequence.

        Returns:
            [seq_len] tensor of weights in (0, 1].
        """
        positions = torch.arange(seq_len, dtype=torch.float32)
        weights = 1.0 / (1.0 + positions)
        return weights

    def augment(
        self,
        src_ids: Tensor,
        importance_scores: Tensor,
    ) -> Tensor:
        """Boost importance of first lead_frac tokens by position bias.

        Args:
            src_ids:           [B, T_src] source token ids (unused — kept for
                               interface consistency; shape determines T_src).
            importance_scores: [B, T_src] initial importance scores.

        Returns:
            [B, T_src] augmented importance scores (same shape).
        """
        B, T = importance_scores.shape
        lead_len = max(1, int(math.ceil(self.lead_frac * T)))

        bias = self.compute_position_bias(T).to(importance_scores.device)
        # Only apply bias to the lead portion; tail stays unchanged.
        lead_bias = bias[:lead_len]  # [lead_len]
        tail_ones = torch.ones(
            T - lead_len, dtype=importance_scores.dtype, device=importance_scores.device
        )
        multiplier = torch.cat([lead_bias.to(importance_scores.dtype), tail_ones])
        # Broadcast over batch
        augmented = importance_scores * multiplier.unsqueeze(0)  # [B, T]
        return augmented


# ---------------------------------------------------------------------------
# SummarizationConfig
# ---------------------------------------------------------------------------


@dataclass
class SummarizationConfig:
    """Hyperparameters for the summarization training pipeline."""

    coverage_lambda: float = 1.0
    length_penalty: float = 0.6
    alpha: float = 0.5
    n_extract_sentences: int = 3
    lead_frac: float = 0.3
    lr: float = 1e-4
    d_model: int = 32
    vocab_size: int = 64
