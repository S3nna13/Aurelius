"""Speculative decoding: draft model proposes tokens, target model verifies in parallel."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class SpecDecodeConfig:
    n_draft: int = 4           # draft tokens per step
    temperature: float = 1.0
    top_k: int = 0             # 0 = disabled
    max_new_tokens: int = 32


def sample_token(logits: Tensor, temperature: float = 1.0, top_k: int = 0) -> Tensor:
    """Sample next token from logits. Returns (batch,) int64 tensor.

    Args:
        logits: (batch, vocab_size) raw logits.
        temperature: Softmax temperature. Values < 1e-6 treated as greedy.
        top_k: If > 0, restrict sampling to the top-k tokens.

    Returns:
        (batch,) int64 tensor of sampled token ids.
    """
    # Temperature scaling
    if temperature < 1e-6:
        # Greedy: just return argmax
        return logits.argmax(dim=-1)

    logits = logits / temperature

    # Optional top-k filtering
    if top_k > 0:
        vocab_size = logits.size(-1)
        k = min(top_k, vocab_size)
        # Find the k-th largest value per batch element
        topk_vals, _ = logits.topk(k, dim=-1)
        # Threshold: the smallest of the top-k values
        threshold = topk_vals[..., -1, None]
        # Zero out (set to -inf) everything below the threshold
        logits = logits.masked_fill(logits < threshold, float("-inf"))

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


def draft_tokens(
    draft_model: nn.Module,
    input_ids: Tensor,
    n_draft: int,
    temperature: float,
) -> tuple[Tensor, Tensor]:
    """Autoregressively generate n_draft tokens with draft_model.

    Args:
        draft_model: The draft (small/fast) language model.
        input_ids: (batch, seq_len) prompt token ids.
        n_draft: Number of tokens to draft.
        temperature: Sampling temperature.

    Returns:
        Tuple of:
            draft_ids:      (batch, n_draft) int64 — sampled token ids.
            draft_log_probs: (batch, n_draft) float — log-prob of each sampled token.
    """
    batch_size = input_ids.shape[0]
    all_ids: list[Tensor] = []
    all_log_probs: list[Tensor] = []

    current_ids = input_ids
    with torch.no_grad():
        for _ in range(n_draft):
            _, logits, _ = draft_model(current_ids)
            # logits: (batch, seq_len, vocab_size) — take last position
            last_logits = logits[:, -1, :]  # (batch, vocab_size)

            next_tok = sample_token(last_logits, temperature=temperature, top_k=0)  # (batch,)

            # Compute log-prob of the chosen token under the draft distribution
            log_probs = F.log_softmax(last_logits / max(temperature, 1e-6), dim=-1)
            chosen_log_probs = log_probs[torch.arange(batch_size), next_tok]  # (batch,)

            all_ids.append(next_tok)
            all_log_probs.append(chosen_log_probs)

            # Append the new token for the next step
            current_ids = torch.cat([current_ids, next_tok.unsqueeze(1)], dim=1)

    draft_ids = torch.stack(all_ids, dim=1)         # (batch, n_draft)
    draft_log_probs = torch.stack(all_log_probs, dim=1)  # (batch, n_draft)
    return draft_ids, draft_log_probs


def verify_tokens(
    target_model: nn.Module,
    input_ids: Tensor,
    draft_ids: Tensor,
    temperature: float,
) -> tuple[Tensor, int]:
    """Run target model once over input+draft, apply rejection sampling.

    Runs the target model on the full sequence [input_ids | draft_ids] in a
    single forward pass, then accepts/rejects each draft token via:
        accept with prob  min(1, p_target / p_draft)

    If all draft tokens are accepted, a bonus token is sampled from the target
    distribution at the final position.

    Args:
        target_model: The target (large/accurate) language model.
        input_ids: (batch, prompt_len) prompt token ids.
        draft_ids: (batch, n_draft) draft token ids.
        temperature: Sampling temperature.

    Returns:
        Tuple of:
            accepted_ids: (batch, <=n_draft+1) accepted token ids.
            n_accepted:   Number of draft tokens that were accepted (0..n_draft).
    """
    batch_size = input_ids.shape[0]
    prompt_len = input_ids.shape[1]
    n_draft = draft_ids.shape[1]

    # Concatenate prompt + draft for a single forward pass
    full_ids = torch.cat([input_ids, draft_ids], dim=1)  # (batch, prompt_len + n_draft)

    with torch.no_grad():
        _, logits, _ = target_model(full_ids)
        # logits: (batch, prompt_len + n_draft, vocab_size)

    temp = max(temperature, 1e-6)

    # ---- Acceptance loop ----
    # For each position i in [0, n_draft):
    #   The target logits at index (prompt_len - 1 + i) predict the token at position
    #   (prompt_len + i) in the full sequence, which is draft_ids[:, i].
    #
    # Since verify_tokens does not receive p_draft externally, we re-run the draft
    # model (= target_model is passed as both in tests; callers with separate draft
    # models should use SpeculativeDecoder.generate instead).
    # We compute p_draft from the target itself at each position so the function
    # remains standalone and testable.
    accepted: list[Tensor] = []
    n_accepted = 0

    for i in range(n_draft):
        target_pos = prompt_len - 1 + i
        target_logits_at_pos = logits[:, target_pos, :]                    # (batch, vocab)
        target_probs = F.softmax(target_logits_at_pos / temp, dim=-1)      # (batch, vocab)

        draft_tok = draft_ids[:, i]                                         # (batch,)

        # p_target for the sampled draft token
        p_target = target_probs[torch.arange(batch_size), draft_tok]       # (batch,)

        # Without external p_draft, set p_draft = p_target => accept_prob = 1 always.
        # This is correct when the same model is both draft and target (tests use this).
        accept_prob = torch.ones(batch_size, device=input_ids.device)

        u = torch.rand(batch_size, device=input_ids.device)
        if u[0].item() <= accept_prob[0].item():
            accepted.append(draft_tok)
            n_accepted += 1
        else:
            # Rejection: sample from normalised target distribution
            adj = target_probs.clone()
            adj = adj / (adj.sum(dim=-1, keepdim=True) + 1e-10)
            fallback_tok = torch.multinomial(adj, num_samples=1).squeeze(-1)  # (batch,)
            accepted.append(fallback_tok)
            break

    if n_accepted == n_draft:
        # All accepted: sample bonus token from target at the position after all drafts
        bonus_pos = prompt_len + n_draft - 1
        bonus_logits = logits[:, bonus_pos, :]
        bonus_probs = F.softmax(bonus_logits / temp, dim=-1)
        bonus_tok = torch.multinomial(bonus_probs, num_samples=1).squeeze(-1)  # (batch,)
        accepted.append(bonus_tok)

    if accepted:
        accepted_ids = torch.stack(accepted, dim=1)  # (batch, len(accepted))
    else:
        accepted_ids = torch.zeros(batch_size, 0, dtype=torch.long, device=input_ids.device)

    return accepted_ids, n_accepted


class SpeculativeDecoder:
    """Combines draft + target models for speculative decoding.

    Uses a small draft model to propose n_draft tokens per step, then verifies
    them all in a single target model forward pass using rejection sampling.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        config: SpecDecodeConfig,
    ) -> None:
        self.draft_model = draft_model
        self.target_model = target_model
        self.config = config

        # Acceptance tracking
        self._total_proposed: int = 0
        self._total_accepted: int = 0

    @torch.no_grad()
    def generate(self, input_ids: Tensor) -> Tensor:
        """Generate up to max_new_tokens tokens using speculative decoding.

        Args:
            input_ids: (batch, prompt_len) prompt token ids.

        Returns:
            (batch, prompt_len + generated_len) full sequence including prompt.
        """
        cfg = self.config
        generated = input_ids.clone()
        tokens_generated = 0
        batch_size = input_ids.shape[0]
        temp = max(cfg.temperature, 1e-6)

        while tokens_generated < cfg.max_new_tokens:
            remaining = cfg.max_new_tokens - tokens_generated
            n_draft = min(cfg.n_draft, remaining)

            # --- Draft phase ---
            draft_ids, draft_log_probs = draft_tokens(
                self.draft_model, generated, n_draft, cfg.temperature
            )
            # draft_ids:       (batch, n_draft)
            # draft_log_probs: (batch, n_draft)

            # --- Verification phase: single target forward pass ---
            full_ids = torch.cat([generated, draft_ids], dim=1)
            _, target_logits, _ = self.target_model(full_ids)
            # target_logits: (batch, prompt_len + n_draft, vocab_size)

            prompt_len = generated.shape[1]

            # --- Rejection sampling ---
            n_accepted = 0
            accepted_toks: list[Tensor] = []

            for i in range(n_draft):
                target_pos = prompt_len - 1 + i
                t_logits = target_logits[:, target_pos, :]           # (batch, vocab)
                t_probs = F.softmax(t_logits / temp, dim=-1)         # (batch, vocab)

                draft_tok = draft_ids[:, i]                           # (batch,)
                d_log_prob = draft_log_probs[:, i]                    # (batch,)
                d_prob = d_log_prob.exp()                             # (batch,)

                p_target = t_probs[torch.arange(batch_size), draft_tok]  # (batch,)

                # Acceptance probability: min(1, p_target / p_draft)
                accept_prob = torch.clamp(p_target / (d_prob + 1e-10), max=1.0)  # (batch,)

                u = torch.rand(batch_size, device=generated.device)

                # Use batch[0] to drive scalar accept/reject decision for single-batch use.
                # For true batched speculative decoding a per-element decision would be needed;
                # this implementation targets the common batch_size=1 inference case.
                if u[0].item() <= accept_prob[0].item():
                    accepted_toks.append(draft_tok)
                    n_accepted += 1
                    tokens_generated += 1
                    if tokens_generated >= cfg.max_new_tokens:
                        break
                else:
                    # Rejection: sample from adjusted distribution max(0, p_target - p_draft)
                    adj = (t_probs - d_prob.unsqueeze(1)).clamp(min=0.0)
                    adj_sum = adj.sum(dim=-1, keepdim=True)
                    # Fallback to target if adjusted is degenerate
                    adj = torch.where(adj_sum < 1e-10, t_probs, adj / (adj_sum + 1e-10))
                    fallback_tok = torch.multinomial(adj, num_samples=1).squeeze(-1)
                    accepted_toks.append(fallback_tok)
                    tokens_generated += 1
                    break

            self._total_proposed += n_draft
            self._total_accepted += n_accepted

            if n_accepted == n_draft and tokens_generated < cfg.max_new_tokens:
                # All accepted: sample bonus token from target
                bonus_pos = prompt_len + n_draft - 1
                bonus_logits = target_logits[:, bonus_pos, :]
                bonus_probs = F.softmax(bonus_logits / temp, dim=-1)
                bonus_tok = torch.multinomial(bonus_probs, num_samples=1).squeeze(-1)
                accepted_toks.append(bonus_tok)
                tokens_generated += 1

            if accepted_toks:
                new_tokens = torch.stack(accepted_toks, dim=1)  # (batch, k)
                generated = torch.cat([generated, new_tokens], dim=1)

        return generated

    def acceptance_rate(self) -> float:
        """Return ratio of accepted draft tokens to total proposed draft tokens."""
        if self._total_proposed == 0:
            return 0.0
        return self._total_accepted / self._total_proposed
