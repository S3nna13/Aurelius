"""Cross-encoder reranker for Aurelius retrieval pipeline.

Implements the Nogueira & Cho (2019) "Passage Re-ranking with BERT"
(arXiv:1901.04085) pattern: a small transformer consumes a concatenated
``[CLS] query [SEP] doc`` token sequence and a scalar head over the
``[CLS]`` position produces a relevance logit.

This module provides *architecture* + an inference harness only; training
and tokenization land in later cycles. It is deliberately self-contained:
attention and normalization are reimplemented with pure ``torch`` native
modules to avoid any coupling to the frozen ``src.model`` core (RMSNorm,
rotary attention, etc.). LayerNorm is used instead of RMSNorm.

Public surface:
    - CrossEncoderConfig
    - CrossEncoderReranker
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class CrossEncoderConfig:
    """Configuration for :class:`CrossEncoderReranker`.

    Defaults target a pocket-sized model suitable for unit tests and initial
    fine-tuning experiments; production configurations will override these
    via YAML.
    """

    vocab_size: int = 256
    d_model: int = 64
    n_layers: int = 2
    n_heads: int = 4
    d_ff: int = 128
    max_seq_len: int = 128
    dropout: float = 0.0
    sep_token_id: int = 1

    def __post_init__(self) -> None:
        if self.vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.n_layers <= 0:
            raise ValueError("n_layers must be positive")
        if self.n_heads <= 0:
            raise ValueError("n_heads must be positive")
        if self.d_model % self.n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")
        if self.d_ff <= 0:
            raise ValueError("d_ff must be positive")
        if self.max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError("dropout must be in [0, 1)")
        if self.sep_token_id < 0 or self.sep_token_id >= self.vocab_size:
            raise ValueError("sep_token_id must index into vocab")


class _MultiHeadSelfAttention(nn.Module):
    """Pure-torch multi-head self-attention.

    Reimplemented locally (rather than imported from ``src.model``) to keep
    the reranker decoupled from the frozen core transformer.
    """

    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)
        self.dropout_p = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        qkv = self.qkv_proj(x)  # [B, S, 3*D]
        q, k, v = qkv.chunk(3, dim=-1)
        # Reshape to [B, H, S, Dh]
        q = q.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(b, s, self.n_heads, self.d_head).transpose(1, 2)
        # Bidirectional (no causal mask) — reranker is encoder-style.
        attn = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_head)
        attn = F.softmax(attn, dim=-1)
        if self.dropout_p > 0.0 and self.training:
            attn = F.dropout(attn, p=self.dropout_p)
        out = torch.matmul(attn, v)  # [B, H, S, Dh]
        out = out.transpose(1, 2).contiguous().view(b, s, self.d_model)
        return self.out_proj(out)


class _EncoderBlock(nn.Module):
    """Pre-LN transformer encoder block (self-attn + FFN)."""

    def __init__(self, cfg: CrossEncoderConfig) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.d_model)
        self.attn = _MultiHeadSelfAttention(cfg.d_model, cfg.n_heads, cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.d_model)
        self.ff = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_ff, bias=True),
            nn.GELU(),
            nn.Linear(cfg.d_ff, cfg.d_model, bias=True),
        )
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.dropout(self.attn(self.norm1(x)))
        x = x + self.dropout(self.ff(self.norm2(x)))
        return x


class CrossEncoderReranker(nn.Module):
    """Small BERT-style cross-encoder producing a scalar relevance logit.

    Input is a single concatenated sequence of token IDs of the form
    ``[CLS=0] query_ids [SEP] doc_ids`` (see :meth:`score_pair` /
    :meth:`rerank` for construction helpers). The forward returns a
    ``[B]``-shaped logit tensor.
    """

    CLS_TOKEN_ID: int = 0

    def __init__(self, config: CrossEncoderConfig) -> None:
        super().__init__()
        if not isinstance(config, CrossEncoderConfig):
            raise TypeError("config must be a CrossEncoderConfig instance")
        self.config = config
        self.tok_embed = nn.Embedding(config.vocab_size, config.d_model)
        self.pos_embed = nn.Embedding(config.max_seq_len, config.d_model)
        self.input_dropout = nn.Dropout(config.dropout)
        self.layers = nn.ModuleList(
            [_EncoderBlock(config) for _ in range(config.n_layers)]
        )
        self.final_norm = nn.LayerNorm(config.d_model)
        self.score_head = nn.Linear(config.d_model, 1, bias=True)
        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Compute relevance logits for a batch of (query, doc) sequences.

        Args:
            input_ids: LongTensor of shape ``[B, S]`` with
                ``S <= config.max_seq_len``. Callers are expected to have
                prepended ``[CLS]`` (token id 0) and separated query / doc
                with ``config.sep_token_id``.

        Returns:
            FloatTensor of shape ``[B]`` — raw logits (apply sigmoid
            externally for probabilities).
        """
        if input_ids.dim() != 2:
            raise ValueError(
                f"input_ids must be 2D [B, S]; got shape {tuple(input_ids.shape)}"
            )
        b, s = input_ids.shape
        if s == 0:
            raise ValueError("input_ids sequence length must be >= 1")
        if s > self.config.max_seq_len:
            raise ValueError(
                f"sequence length {s} exceeds max_seq_len {self.config.max_seq_len}"
            )
        if input_ids.dtype not in (torch.long, torch.int64, torch.int32):
            raise ValueError("input_ids must be an integer tensor")
        if torch.any(input_ids < 0) or torch.any(input_ids >= self.config.vocab_size):
            raise ValueError("input_ids contain ids outside [0, vocab_size)")

        positions = torch.arange(s, device=input_ids.device).unsqueeze(0).expand(b, s)
        x = self.tok_embed(input_ids) + self.pos_embed(positions)
        x = self.input_dropout(x)
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        cls_repr = x[:, 0, :]  # [B, D] — [CLS] position
        logits = self.score_head(cls_repr).squeeze(-1)  # [B]
        return logits

    # ------------------------------------------------------------------
    # Inference helpers
    # ------------------------------------------------------------------

    def _build_sequence(
        self, query_ids: list[int], doc_ids: list[int]
    ) -> torch.Tensor:
        if not isinstance(query_ids, (list, tuple)) or not isinstance(
            doc_ids, (list, tuple)
        ):
            raise TypeError("query_ids and doc_ids must be lists of ints")
        # [CLS] + query + [SEP] + doc
        seq = [self.CLS_TOKEN_ID, *query_ids, self.config.sep_token_id, *doc_ids]
        if len(seq) > self.config.max_seq_len:
            raise ValueError(
                f"combined sequence length {len(seq)} exceeds max_seq_len "
                f"{self.config.max_seq_len}; caller must truncate before reranking"
            )
        for tid in seq:
            if not isinstance(tid, int):
                raise TypeError("token ids must be Python ints")
            if tid < 0 or tid >= self.config.vocab_size:
                raise ValueError(f"token id {tid} outside [0, vocab_size)")
        device = next(self.parameters()).device
        return torch.tensor(seq, dtype=torch.long, device=device).unsqueeze(0)

    def _run_inference(self, seq: torch.Tensor) -> torch.Tensor:
        """Run a forward pass under no-grad + eval mode, restoring state after."""
        was_training = self.training
        self.train(False)
        try:
            with torch.no_grad():
                return self.forward(seq)
        finally:
            self.train(was_training)

    def score_pair(self, query_ids: list[int], doc_ids: list[int]) -> float:
        """Return a scalar relevance score for a single (query, doc) pair."""
        seq = self._build_sequence(query_ids, doc_ids)
        logit = self._run_inference(seq)
        return float(logit.item())

    def rerank(
        self,
        query_ids: list[int],
        list_of_doc_ids: list[list[int]],
    ) -> list[tuple[int, float]]:
        """Score and rank a list of candidate documents for a single query.

        Args:
            query_ids: Token IDs for the query.
            list_of_doc_ids: List of token-ID lists, one per candidate doc.

        Returns:
            List of ``(doc_idx, score)`` tuples sorted by score descending.
        """
        if not isinstance(list_of_doc_ids, (list, tuple)):
            raise TypeError("list_of_doc_ids must be a list of lists of ints")
        if len(list_of_doc_ids) == 0:
            return []
        scored: list[tuple[int, float]] = []
        for idx, doc_ids in enumerate(list_of_doc_ids):
            scored.append((idx, self.score_pair(query_ids, doc_ids)))
        scored.sort(key=lambda pair: pair[1], reverse=True)
        return scored


__all__ = ["CrossEncoderConfig", "CrossEncoderReranker"]
