"""Speculative decoding: draft model proposes tokens, target model verifies in parallel.

Classic speculative decoding following:
  - Leviathan et al. 2023 ("Fast Inference from Transformers via Speculative Decoding")
  - Chen et al. 2023 ("Accelerating Large Language Model Decoding with Speculative Sampling")

Public API
----------
SpeculativeConfig   — configuration dataclass
DraftModel          — wraps a callable model_fn, autoregressively proposes K tokens
SpeculativeVerifier — implements rejection-sampling accept/reject step
SpeculativeDecoder  — full draft + verify loop
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding."""

    n_draft_tokens: int = 5
    temperature: float = 1.0
    top_p: float = 1.0  # nucleus sampling for target; 1.0 means no filtering
    max_new_tokens: int = 128


# ---------------------------------------------------------------------------
# Draft Model
# ---------------------------------------------------------------------------


class DraftModel:
    """Wraps a callable draft model function for autoregressive token proposal.

    Parameters
    ----------
    model_fn:
        Callable that accepts ``input_ids: LongTensor(1, T)`` and returns
        ``logits: FloatTensor(1, T, V)``.
    vocab_size:
        Vocabulary size ``V``.
    """

    def __init__(
        self,
        model_fn: Callable[[Tensor], Tensor],
        vocab_size: int,
    ) -> None:
        self.model_fn = model_fn
        self.vocab_size = vocab_size

    def autoregressive_draft(
        self,
        input_ids: Tensor,
        n_tokens: int,
    ) -> tuple[Tensor, Tensor]:
        """Greedily generate ``n_tokens`` tokens from the draft model.

        Parameters
        ----------
        input_ids:
            ``(1, T)`` int64 prompt tensor.
        n_tokens:
            Number of tokens to draft.

        Returns
        -------
        token_ids : ``LongTensor(n_tokens,)`` — greedy draft token ids.
        logits_per_step : ``FloatTensor(n_tokens, V)`` — raw logits at each step.
        """
        token_ids_list: list[Tensor] = []
        logits_list: list[Tensor] = []

        current = input_ids  # (1, T)

        with torch.no_grad():
            for _ in range(n_tokens):
                logits = self.model_fn(current)  # (1, T, V)
                last_logits = logits[0, -1, :]  # (V,)

                # Greedy selection
                next_tok = last_logits.argmax(dim=-1)  # scalar

                token_ids_list.append(next_tok)
                logits_list.append(last_logits)

                current = torch.cat(
                    [current, next_tok.unsqueeze(0).unsqueeze(0)], dim=1
                )  # (1, T+1)

        token_ids = torch.stack(token_ids_list, dim=0)  # (n_tokens,)
        logits_per_step = torch.stack(logits_list, dim=0)  # (n_tokens, V)
        return token_ids, logits_per_step


# ---------------------------------------------------------------------------
# Speculative Verifier
# ---------------------------------------------------------------------------


class SpeculativeVerifier:
    """Implements the rejection sampling accept/reject criterion.

    Parameters
    ----------
    vocab_size:
        Vocabulary size ``V``.
    temperature:
        Temperature applied when sampling from distributions.
    """

    def __init__(self, vocab_size: int, temperature: float = 1.0) -> None:
        self.vocab_size = vocab_size
        self.temperature = temperature

    def verify(
        self,
        draft_ids: Tensor,
        draft_probs: Tensor,
        target_probs: Tensor,
    ) -> tuple[Tensor, int]:
        """Apply rejection sampling to accept/reject draft tokens.

        For each position ``i``:
        * Compute ``accept_prob = min(1, p_target[i, draft_ids[i]] / p_draft[i, draft_ids[i]])``.
        * Draw ``u ~ Uniform(0, 1)``.
        * If ``u <= accept_prob``: accept the draft token.
        * Else: sample corrected token from ``q = max(0, p_t - p_d) / Z``; stop.

        If all draft tokens are accepted, a bonus token is sampled from the
        target distribution at the last position.

        Parameters
        ----------
        draft_ids:
            ``LongTensor(K,)`` — draft token indices.
        draft_probs:
            ``FloatTensor(K, V)`` — draft probability distributions.
        target_probs:
            ``FloatTensor(K, V)`` — target probability distributions.

        Returns
        -------
        accepted_tokens : ``LongTensor(m,)`` — accepted token ids; includes
            corrected token on rejection or bonus token when all accepted.
        n_accepted : int — number of original draft tokens accepted (0..K).
        """
        K = draft_ids.shape[0]
        accepted_list: list[Tensor] = []
        n_accepted = 0

        for i in range(K):
            tok = draft_ids[i]  # scalar
            p_d = draft_probs[i, tok]  # scalar
            p_t = target_probs[i, tok]  # scalar

            # Acceptance probability
            accept_prob = torch.clamp(p_t / (p_d + 1e-10), max=1.0)
            u = torch.rand(1, device=draft_ids.device).squeeze()

            if u.item() <= accept_prob.item():
                accepted_list.append(tok)
                n_accepted += 1
            else:
                # Corrected distribution: q = max(0, p_target - p_draft) / Z
                q = (target_probs[i] - draft_probs[i]).clamp(min=0.0)
                q_sum = q.sum()
                if q_sum.item() < 1e-10:
                    # Degenerate fallback: sample from target
                    q = target_probs[i].clone()
                    q_sum = q.sum()
                q = q / (q_sum + 1e-10)
                corrected = torch.multinomial(q, num_samples=1).squeeze()
                accepted_list.append(corrected)
                break
        else:
            # All K draft tokens accepted — sample bonus from last target dist
            bonus = torch.multinomial(target_probs[K - 1], num_samples=1).squeeze()
            accepted_list.append(bonus)

        if accepted_list:
            accepted_tokens = torch.stack(accepted_list, dim=0)  # (m,)
        else:
            accepted_tokens = torch.zeros(0, dtype=torch.long, device=draft_ids.device)

        return accepted_tokens, n_accepted

    def sample_from_logits(self, logits: Tensor) -> int:
        """Sample a token from logits using temperature softmax.

        Parameters
        ----------
        logits : ``(V,)`` raw logits.

        Returns
        -------
        int — sampled token id in ``[0, V)``.
        """
        temp = max(float(self.temperature), 1e-8)
        probs = F.softmax(logits / temp, dim=-1)  # (V,)
        tok = torch.multinomial(probs, num_samples=1).squeeze()
        return int(tok.item())


# ---------------------------------------------------------------------------
# Full Speculative Decoder
# ---------------------------------------------------------------------------


class SpeculativeDecoder:
    """Full speculative decoding loop.

    Parameters
    ----------
    draft_model_fn:
        Callable ``(1, T) -> (1, T, V)`` for the small draft model.
    target_model_fn:
        Callable ``(1, T) -> (1, T, V)`` for the large target model.
    vocab_size:
        Vocabulary size ``V``.
    config:
        :class:`SpeculativeConfig` instance.
    """

    def __init__(
        self,
        draft_model_fn: Callable[[Tensor], Tensor],
        target_model_fn: Callable[[Tensor], Tensor],
        vocab_size: int,
        config: SpeculativeConfig,
    ) -> None:
        self.draft_model = DraftModel(draft_model_fn, vocab_size)
        self.target_model_fn = target_model_fn
        self.vocab_size = vocab_size
        self.config = config
        self.verifier = SpeculativeVerifier(vocab_size, config.temperature)

    def generate(
        self,
        prompt_ids: Tensor,
        max_new_tokens: int,
    ) -> Tensor:
        """Generate tokens using speculative decoding.

        Parameters
        ----------
        prompt_ids:
            ``LongTensor(1, T)`` prompt tensor.
        max_new_tokens:
            Maximum number of new tokens to generate.

        Returns
        -------
        ``LongTensor(max_new_tokens,)`` — the generated token ids (not including
        the prompt), truncated to exactly ``max_new_tokens``.
        """
        cfg = self.config
        temp = max(float(cfg.temperature), 1e-8)
        K = cfg.n_draft_tokens

        context = prompt_ids.clone()  # (1, T)
        generated: list[Tensor] = []  # list of 1-D token tensors
        tokens_so_far = 0

        with torch.no_grad():
            while tokens_so_far < max_new_tokens:
                remaining = max_new_tokens - tokens_so_far
                k = min(K, remaining)

                # --- Draft phase ---
                draft_ids, draft_logits = self.draft_model.autoregressive_draft(
                    context, k
                )  # (k,), (k, V)
                draft_probs = F.softmax(draft_logits / temp, dim=-1)  # (k, V)

                # --- Target scoring phase (one parallel forward pass) ---
                # Append draft tokens to context for a single forward pass
                full_ids = torch.cat([context, draft_ids.unsqueeze(0)], dim=1)  # (1, T+k)
                target_logits_all = self.target_model_fn(full_ids)  # (1, T+k, V)

                # The target prediction at position (T-1 + i) predicts draft_ids[i]
                # for i in 0..k-1.
                prompt_len = context.shape[1]
                # Extract target logits at the k draft positions
                target_logits_draft = target_logits_all[
                    0, prompt_len - 1 : prompt_len - 1 + k, :
                ]  # (k, V)
                target_probs_draft = F.softmax(target_logits_draft / temp, dim=-1)  # (k, V)

                # Apply top-p (nucleus) filtering to target probs if configured
                if cfg.top_p < 1.0:
                    target_probs_draft = _apply_top_p(target_probs_draft, cfg.top_p)

                # --- Verification phase ---
                accepted_tokens, n_accepted = self.verifier.verify(
                    draft_ids, draft_probs, target_probs_draft
                )
                # accepted_tokens: (m,), m >= 1 (always at least corrected/bonus)

                # Truncate to remaining budget
                n_to_add = min(accepted_tokens.shape[0], remaining)
                new_toks = accepted_tokens[:n_to_add]

                generated.append(new_toks)
                tokens_so_far += n_to_add

                # Advance context
                context = torch.cat([context, new_toks.unsqueeze(0)], dim=1)  # (1, T + n_to_add)

        if generated:
            result = torch.cat(generated, dim=0)
        else:
            result = torch.zeros(0, dtype=torch.long, device=prompt_ids.device)

        # Always return exactly max_new_tokens tokens
        return result[:max_new_tokens]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _apply_top_p(probs: Tensor, top_p: float) -> Tensor:
    """Apply nucleus (top-p) filtering to a batch of probability distributions.

    Parameters
    ----------
    probs : ``(K, V)`` probability distributions.
    top_p : float in (0, 1].

    Returns
    -------
    Filtered and renormalized ``(K, V)`` distributions.
    """
    sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # Remove tokens with cumulative probability > top_p
    # (shift by one to keep first token that crosses the threshold)
    sorted_indices_to_remove = cumulative_probs - sorted_probs > top_p
    sorted_probs = sorted_probs.masked_fill(sorted_indices_to_remove, 0.0)

    # Scatter back to original ordering
    filtered = torch.zeros_like(probs)
    filtered.scatter_(-1, sorted_indices, sorted_probs)

    # Renormalize
    filtered = filtered / (filtered.sum(dim=-1, keepdim=True) + 1e-10)
    return filtered
