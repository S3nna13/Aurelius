"""
Sequence-to-Sequence model with encoder-decoder architecture and beam search decoding.

Implements:
  - Seq2SeqEncoder: bidirectional-style transformer encoder (non-causal self-attention)
  - Seq2SeqDecoder: causal decoder with cross-attention + KV cache for beam search
  - Seq2SeqModel: combined encoder-decoder
  - BeamSearchDecoder: standard beam search with length penalty
  - DiverseBeamSearch: grouped beam search with diversity penalty
  - Seq2SeqConfig: dataclass of hyperparameters
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    """Upper-triangular mask: positions j > i are masked (value = -inf)."""
    mask = torch.full((seq_len, seq_len), float("-inf"), device=device)
    mask = torch.triu(mask, diagonal=1)
    return mask  # [T, T]


class _MultiHeadAttention(nn.Module):
    """Scaled dot-product multi-head attention with optional causal mask."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"  # noqa: S101
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, T, d_model] -> [B, n_heads, T, d_head]"""
        B, T, _ = x.shape
        x = x.view(B, T, self.n_heads, self.d_head)
        return x.transpose(1, 2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        """[B, n_heads, T, d_head] -> [B, T, d_model]"""
        B, _, T, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.view(B, T, self.d_model)

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        q = self._split_heads(self.q_proj(query))
        k = self._split_heads(self.k_proj(key))
        v = self._split_heads(self.v_proj(value))

        if kv_cache is not None:
            k_cache, v_cache = kv_cache
            k = torch.cat([k_cache, k], dim=2)
            v = torch.cat([v_cache, v], dim=2)

        new_cache = (k, v)

        scale = math.sqrt(self.d_head)
        scores = torch.matmul(q, k.transpose(-2, -1)) / scale  # [B, H, Tq, Tk]

        if attn_mask is not None:
            scores = scores + attn_mask

        weights = F.softmax(scores, dim=-1)
        out = torch.matmul(weights, v)  # [B, H, Tq, d_head]
        out = self._merge_heads(out)
        out = self.out_proj(out)
        return out, new_cache


class _FFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int | None = None) -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _EncoderBlock(nn.Module):
    """Non-causal transformer block for the encoder."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.self_attn = _MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = _FFN(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out, _ = self.self_attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x


class _DecoderBlock(nn.Module):
    """Decoder block: causal self-attn + cross-attn + FFN."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.self_attn = _MultiHeadAttention(d_model, n_heads)
        self.norm1 = nn.LayerNorm(d_model)
        self.cross_attn = _MultiHeadAttention(d_model, n_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = _FFN(d_model)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        enc_out: torch.Tensor,
        self_attn_mask: torch.Tensor | None = None,
        self_kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
        cross_kv_cache: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[
        torch.Tensor,
        tuple[torch.Tensor, torch.Tensor],
        tuple[torch.Tensor, torch.Tensor],
    ]:
        sa_out, new_self_kv = self.self_attn(
            x,
            x,
            x,
            attn_mask=self_attn_mask,
            kv_cache=self_kv_cache,
        )
        x = self.norm1(x + sa_out)

        if cross_kv_cache is not None:
            ca_out, new_cross_kv = self.cross_attn(
                x,
                enc_out,
                enc_out,
                kv_cache=cross_kv_cache,
            )
        else:
            ca_out, new_cross_kv = self.cross_attn(x, enc_out, enc_out)

        x = self.norm2(x + ca_out)
        x = self.norm3(x + self.ffn(x))
        return x, new_self_kv, new_cross_kv


# ---------------------------------------------------------------------------
# Positional Encoding
# ---------------------------------------------------------------------------


class _SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # [1, max_len, d_model]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


# ---------------------------------------------------------------------------
# Encoder
# ---------------------------------------------------------------------------


class Seq2SeqEncoder(nn.Module):
    """
    Transformer encoder with non-causal (bidirectional) self-attention.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_layers: int = 2,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = _SinusoidalPE(d_model)
        self.layers = nn.ModuleList([_EncoderBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src_ids: [B, T_src]
        Returns:
            enc_out: [B, T_src, d_model]
        """
        x = self.pe(self.embedding(src_ids))
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


# ---------------------------------------------------------------------------
# Decoder
# ---------------------------------------------------------------------------


class Seq2SeqDecoder(nn.Module):
    """
    Transformer decoder with causal self-attention + cross-attention.
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_layers: int = 2,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pe = _SinusoidalPE(d_model)
        self.layers = nn.ModuleList([_DecoderBlock(d_model, n_heads) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(
        self,
        tgt_ids: torch.Tensor,
        enc_out: torch.Tensor,
    ) -> torch.Tensor:
        """
        Full forward pass (training-time, uses causal mask).

        Args:
            tgt_ids: [B, T_tgt]
            enc_out: [B, T_src, d_model]
        Returns:
            logits: [B, T_tgt, vocab_size]
        """
        T_tgt = tgt_ids.size(1)
        causal_mask = _make_causal_mask(T_tgt, tgt_ids.device)
        x = self.pe(self.embedding(tgt_ids))
        for layer in self.layers:
            x, _, _ = layer(x, enc_out, self_attn_mask=causal_mask)
        x = self.norm(x)
        return self.lm_head(x)

    def forward_step(
        self,
        tgt_ids: torch.Tensor,
        enc_out: torch.Tensor,
        cache: list[dict] | None = None,
    ) -> tuple[torch.Tensor, list[dict]]:
        """
        Single-step decode with KV cache.

        Args:
            tgt_ids: [B, 1]
            enc_out: [B, T_src, d_model]
            cache:   list of per-layer dicts, or None on first call
        Returns:
            logits:    [B, 1, vocab_size]
            new_cache: list of per-layer cache dicts
        """
        x = self.embedding(tgt_ids)  # [B, 1, d_model]

        # Determine positional offset from cache
        if cache is not None and cache[0].get("self_kv") is not None:
            offset = cache[0]["self_kv"][0].size(2)
        else:
            offset = 0

        pe_buf: torch.Tensor = self.pe.pe  # type: ignore[attr-defined]
        x = x + pe_buf[:, offset : offset + 1, :]

        new_cache: list[dict] = []
        for i, layer in enumerate(self.layers):
            layer_cache: dict = cache[i] if cache is not None else {}
            self_kv = layer_cache.get("self_kv", None)
            cross_kv = layer_cache.get("cross_kv", None)

            x, new_self_kv, new_cross_kv = layer(
                x,
                enc_out,
                self_attn_mask=None,
                self_kv_cache=self_kv,
                cross_kv_cache=cross_kv,
            )
            new_cache.append({"self_kv": new_self_kv, "cross_kv": new_cross_kv})

        x = self.norm(x)
        logits = self.lm_head(x)  # [B, 1, vocab_size]
        return logits, new_cache


# ---------------------------------------------------------------------------
# Seq2SeqModel
# ---------------------------------------------------------------------------


class Seq2SeqModel(nn.Module):
    """Combined encoder-decoder seq2seq model."""

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_layers: int = 2,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.encoder = Seq2SeqEncoder(d_model, vocab_size, n_layers, n_heads)
        self.decoder = Seq2SeqDecoder(d_model, vocab_size, n_layers, n_heads)

    def forward(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            src_ids: [B, T_src]
            tgt_ids: [B, T_tgt]
        Returns:
            logits: [B, T_tgt, vocab_size]
        """
        enc_out = self.encoder(src_ids)
        return self.decoder(tgt_ids, enc_out)

    def compute_loss(
        self,
        src_ids: torch.Tensor,
        tgt_ids: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Cross-entropy loss.

        Args:
            src_ids: [B, T_src]
            tgt_ids: [B, T_tgt]
            labels:  [B, T_tgt]  (-100 positions are ignored)
        Returns:
            scalar CE loss
        """
        logits = self.forward(src_ids, tgt_ids)  # [B, T_tgt, vocab]
        B, T, V = logits.shape
        return F.cross_entropy(
            logits.reshape(B * T, V),
            labels.reshape(B * T),
            ignore_index=-100,
        )


# ---------------------------------------------------------------------------
# BeamSearchDecoder
# ---------------------------------------------------------------------------


class BeamSearchDecoder:
    """
    Standard beam search decoder.
    """

    def __init__(
        self,
        model: Seq2SeqModel,
        beam_size: int,
        max_len: int,
        bos_id: int = 1,
        eos_id: int = 2,
        length_penalty: float = 0.6,
    ) -> None:
        self.model = model
        self.beam_size = beam_size
        self.max_len = max_len
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.length_penalty = length_penalty

    def decode(self, src_ids: torch.Tensor) -> list[list[int]]:
        """
        Beam-search decode a batch of source sequences.

        Args:
            src_ids: [B, T_src]
        Returns:
            List[B] of token id lists (best beam, excluding BOS)
        """
        self.model.eval()
        with torch.no_grad():
            results: list[list[int]] = []
            B = src_ids.size(0)
            for b in range(B):
                best = self._decode_single(src_ids[b : b + 1])
                results.append(best)
        return results

    def _decode_single(self, src_ids: torch.Tensor) -> list[int]:
        """Decode one example using beam search."""
        device = src_ids.device
        K = self.beam_size

        enc_out = self.model.encoder(src_ids)  # [1, T_src, d_model]
        enc_out_k = enc_out.expand(K, -1, -1)  # [K, T_src, d_model]

        beam_tokens: list[list[int]] = [[] for _ in range(K)]
        beam_scores: list[float] = [0.0] * K
        beam_caches: list[list[dict] | None] = [None] * K
        completed: list[tuple[float, list[int]]] = []

        # --- BOS step ---
        bos_tensor = torch.full((K, 1), self.bos_id, dtype=torch.long, device=device)
        logits, new_cache_batch = self.model.decoder.forward_step(bos_tensor, enc_out_k)
        log_probs = F.log_softmax(logits[:, 0, :], dim=-1)  # [K, vocab]

        topk_scores, topk_ids = log_probs[0].topk(K)
        for i in range(K):
            beam_tokens[i] = [topk_ids[i].item()]
            beam_scores[i] = topk_scores[i].item()
            beam_caches[i] = [
                {
                    "self_kv": (
                        new_cache_batch[li]["self_kv"][0][i : i + 1],
                        new_cache_batch[li]["self_kv"][1][i : i + 1],
                    ),
                    "cross_kv": (
                        new_cache_batch[li]["cross_kv"][0][i : i + 1],
                        new_cache_batch[li]["cross_kv"][1][i : i + 1],
                    ),
                }
                for li in range(len(new_cache_batch))
            ]

        # --- Subsequent steps ---
        for _step in range(1, self.max_len):
            if not beam_tokens:
                break

            last_tokens = torch.tensor(
                [toks[-1] for toks in beam_tokens],
                dtype=torch.long,
                device=device,
            ).unsqueeze(1)  # [K_active, 1]

            K_active = last_tokens.size(0)
            enc_out_active = enc_out.expand(K_active, -1, -1)
            batched_cache = self._batch_caches(beam_caches, device)

            logits, new_cache_batch = self.model.decoder.forward_step(
                last_tokens, enc_out_active, batched_cache
            )
            log_probs = F.log_softmax(logits[:, 0, :], dim=-1)  # [K_active, vocab]

            candidates: list[tuple[float, int, int, list[dict]]] = []
            for bi in range(K_active):
                tk_scores, tk_ids = log_probs[bi].topk(K)
                for tk_score, tk_id in zip(tk_scores.tolist(), tk_ids.tolist()):
                    new_score = beam_scores[bi] + tk_score
                    per_beam_cache = [
                        {
                            "self_kv": (
                                new_cache_batch[li]["self_kv"][0][bi : bi + 1],
                                new_cache_batch[li]["self_kv"][1][bi : bi + 1],
                            ),
                            "cross_kv": (
                                new_cache_batch[li]["cross_kv"][0][bi : bi + 1],
                                new_cache_batch[li]["cross_kv"][1][bi : bi + 1],
                            ),
                        }
                        for li in range(len(new_cache_batch))
                    ]
                    candidates.append((new_score, bi, tk_id, per_beam_cache))

            candidates.sort(key=lambda c: c[0], reverse=True)
            candidates = candidates[:K]

            new_beam_tokens: list[list[int]] = []
            new_beam_scores: list[float] = []
            new_beam_caches: list[list[dict] | None] = []

            for score, bi, tok_id, new_bc in candidates:
                seq = beam_tokens[bi] + [tok_id]
                if tok_id == self.eos_id:
                    completed.append((score, seq))
                else:
                    new_beam_tokens.append(seq)
                    new_beam_scores.append(score)
                    new_beam_caches.append(new_bc)

            beam_tokens = new_beam_tokens
            beam_scores = new_beam_scores
            beam_caches = new_beam_caches

            if len(completed) >= K:
                break

        for bi, seq in enumerate(beam_tokens):
            completed.append((beam_scores[bi], seq))

        if not completed:
            return [self.eos_id]

        ranked = self._score_hypotheses(
            [seq for _, seq in completed],
            [sc for sc, _ in completed],
            self.length_penalty,
        )
        return ranked[0][1]

    def _score_hypotheses(
        self,
        hypotheses: list[list[int]],
        scores: list[float],
        length_penalty: float,
    ) -> list[tuple[float, list[int]]]:
        """
        Normalise scores by length and return sorted list (best first).
        normalised = score / (len(hyp) ^ length_penalty)
        """
        normalised: list[tuple[float, list[int]]] = []
        for hyp, sc in zip(hypotheses, scores):
            length = max(len(hyp), 1)
            normalised.append((sc / (length**length_penalty), hyp))
        normalised.sort(key=lambda x: x[0], reverse=True)
        return normalised

    @staticmethod
    def _batch_caches(
        caches: list[list[dict] | None],
        device: torch.device,
    ) -> list[dict]:
        """Stack per-beam caches into a single batched cache."""
        n_layers = len(caches[0])  # type: ignore[arg-type]
        batched: list[dict] = []
        for li in range(n_layers):
            self_k = torch.cat([c[li]["self_kv"][0] for c in caches], dim=0)  # type: ignore
            self_v = torch.cat([c[li]["self_kv"][1] for c in caches], dim=0)
            cross_k = torch.cat([c[li]["cross_kv"][0] for c in caches], dim=0)
            cross_v = torch.cat([c[li]["cross_kv"][1] for c in caches], dim=0)
            batched.append(
                {
                    "self_kv": (self_k, self_v),
                    "cross_kv": (cross_k, cross_v),
                }
            )
        return batched


# ---------------------------------------------------------------------------
# DiverseBeamSearch
# ---------------------------------------------------------------------------


class DiverseBeamSearch(BeamSearchDecoder):
    """
    Grouped beam search with diversity penalty.

    Each group of beams is penalised for selecting tokens already chosen
    by earlier groups at the same time step.
    """

    def __init__(
        self,
        model: Seq2SeqModel,
        beam_size: int,
        max_len: int,
        bos_id: int = 1,
        eos_id: int = 2,
        length_penalty: float = 0.6,
        n_groups: int = 2,
        diversity_penalty: float = 0.5,
    ) -> None:
        super().__init__(model, beam_size, max_len, bos_id, eos_id, length_penalty)
        self.n_groups = n_groups
        self.diversity_penalty = diversity_penalty

    def decode(self, src_ids: torch.Tensor) -> list[list[list[int]]]:  # type: ignore[override]
        """
        Args:
            src_ids: [B, T_src]
        Returns:
            List[B] of List[n_groups] of token id lists
        """
        self.model.eval()
        with torch.no_grad():
            results: list[list[list[int]]] = []
            B = src_ids.size(0)
            for b in range(B):
                group_results = self._decode_single_diverse(src_ids[b : b + 1])
                results.append(group_results)
        return results

    def _decode_single_diverse(self, src_ids: torch.Tensor) -> list[list[int]]:
        """Decode one example using grouped beam search with diversity penalty."""
        device = src_ids.device
        beams_per_group = max(1, self.beam_size // self.n_groups)

        enc_out = self.model.encoder(src_ids)  # [1, T_src, d_model]

        group_results: list[list[int]] = []
        # Track tokens selected by previous groups at each step
        step_tokens_by_group: dict[int, list[int]] = {}

        for g in range(self.n_groups):
            K = beams_per_group
            enc_out_k = enc_out.expand(K, -1, -1)

            beam_tokens: list[list[int]] = [[] for _ in range(K)]
            beam_scores: list[float] = [0.0] * K
            beam_caches: list[list[dict] | None] = [None] * K
            completed: list[tuple[float, list[int]]] = []

            # BOS step
            bos_tensor = torch.full((K, 1), self.bos_id, dtype=torch.long, device=device)
            logits, new_cache_batch = self.model.decoder.forward_step(bos_tensor, enc_out_k)
            log_probs = F.log_softmax(logits[:, 0, :], dim=-1)  # [K, vocab]

            # Apply diversity penalty at step 0 for tokens from earlier groups
            prev_tokens_step0 = step_tokens_by_group.get(0, [])
            if prev_tokens_step0:
                penalty = torch.zeros(log_probs.size(-1), device=device)
                for tok in prev_tokens_step0:
                    penalty[tok] += self.diversity_penalty
                log_probs = log_probs - penalty.unsqueeze(0)

            topk_scores, topk_ids = log_probs[0].topk(K)
            for i in range(K):
                beam_tokens[i] = [topk_ids[i].item()]
                beam_scores[i] = topk_scores[i].item()
                beam_caches[i] = [
                    {
                        "self_kv": (
                            new_cache_batch[li]["self_kv"][0][i : i + 1],
                            new_cache_batch[li]["self_kv"][1][i : i + 1],
                        ),
                        "cross_kv": (
                            new_cache_batch[li]["cross_kv"][0][i : i + 1],
                            new_cache_batch[li]["cross_kv"][1][i : i + 1],
                        ),
                    }
                    for li in range(len(new_cache_batch))
                ]

            # Record step-0 tokens chosen by this group for subsequent groups
            step_0_chosen = [topk_ids[i].item() for i in range(K)]
            existing = step_tokens_by_group.get(0, [])
            step_tokens_by_group[0] = existing + step_0_chosen

            for step in range(1, self.max_len):
                if not beam_tokens:
                    break

                last_tokens = torch.tensor(
                    [toks[-1] for toks in beam_tokens],
                    dtype=torch.long,
                    device=device,
                ).unsqueeze(1)

                K_active = last_tokens.size(0)
                enc_out_active = enc_out.expand(K_active, -1, -1)
                batched_cache = self._batch_caches(beam_caches, device)

                logits, new_cache_batch = self.model.decoder.forward_step(
                    last_tokens, enc_out_active, batched_cache
                )
                log_probs = F.log_softmax(logits[:, 0, :], dim=-1)

                # Diversity penalty from earlier groups' tokens at this step
                prev_step_tokens = step_tokens_by_group.get(step, [])
                if prev_step_tokens:
                    penalty = torch.zeros(log_probs.size(-1), device=device)
                    for tok in prev_step_tokens:
                        penalty[tok] += self.diversity_penalty
                    log_probs = log_probs - penalty.unsqueeze(0)

                candidates: list[tuple[float, int, int, list[dict]]] = []
                for bi in range(K_active):
                    tk_scores, tk_ids = log_probs[bi].topk(K)
                    for tk_score, tk_id in zip(tk_scores.tolist(), tk_ids.tolist()):
                        new_score = beam_scores[bi] + tk_score
                        per_beam_cache = [
                            {
                                "self_kv": (
                                    new_cache_batch[li]["self_kv"][0][bi : bi + 1],
                                    new_cache_batch[li]["self_kv"][1][bi : bi + 1],
                                ),
                                "cross_kv": (
                                    new_cache_batch[li]["cross_kv"][0][bi : bi + 1],
                                    new_cache_batch[li]["cross_kv"][1][bi : bi + 1],
                                ),
                            }
                            for li in range(len(new_cache_batch))
                        ]
                        candidates.append((new_score, bi, tk_id, per_beam_cache))

                candidates.sort(key=lambda c: c[0], reverse=True)
                candidates = candidates[:K]

                # Record chosen tokens for next groups
                step_chosen = [c[2] for c in candidates]
                existing_step = step_tokens_by_group.get(step, [])
                step_tokens_by_group[step] = existing_step + step_chosen

                new_beam_tokens: list[list[int]] = []
                new_beam_scores: list[float] = []
                new_beam_caches: list[list[dict] | None] = []

                for score, bi, tok_id, new_bc in candidates:
                    seq = beam_tokens[bi] + [tok_id]
                    if tok_id == self.eos_id:
                        completed.append((score, seq))
                    else:
                        new_beam_tokens.append(seq)
                        new_beam_scores.append(score)
                        new_beam_caches.append(new_bc)

                beam_tokens = new_beam_tokens
                beam_scores = new_beam_scores
                beam_caches = new_beam_caches

                if len(completed) >= K:
                    break

            for bi, seq in enumerate(beam_tokens):
                completed.append((beam_scores[bi], seq))

            if not completed:
                group_results.append([self.eos_id])
                continue

            ranked = self._score_hypotheses(
                [seq for _, seq in completed],
                [sc for sc, _ in completed],
                self.length_penalty,
            )
            group_results.append(ranked[0][1])

        return group_results


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class Seq2SeqConfig:
    d_model: int = 32
    vocab_size: int = 64
    n_layers: int = 2
    n_heads: int = 4
    beam_size: int = 3
    max_len: int = 16
    length_penalty: float = 0.6
    bos_id: int = 1
    eos_id: int = 2
