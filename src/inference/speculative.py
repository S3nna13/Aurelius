"""Speculative decoding (Leviathan et al., 2023 - arXiv:2211.17192).

Speeds up autoregressive generation by using a small draft model to propose
K tokens, then verifying all K with one target model forward pass.

Expected speedup: beta * K + 1 target calls per K+1 tokens, where beta is
the acceptance rate (how well draft matches target distribution).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class SpeculativeConfig:
    K: int = 4  # draft tokens per speculation round
    temperature: float = 1.0  # sampling temperature (applied to both models)
    top_p: float = 0.9  # nucleus sampling threshold
    max_new_tokens: int = 256
    eos_token_id: int | None = None


def _sample_from_logits(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    """Sample one token from logits with temperature and top-p.

    Args:
        logits: (vocab_size,) — raw logits for one position.

    Returns:
        Scalar tensor — sampled token id.
    """
    if temperature != 1.0:
        logits = logits / temperature

    probs = F.softmax(logits, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        # Remove tokens below the top-p threshold
        sorted_probs[cumulative - sorted_probs > top_p] = 0.0
        sorted_probs /= sorted_probs.sum()
        probs = torch.zeros_like(probs).scatter_(0, sorted_indices, sorted_probs)

    return torch.multinomial(probs, 1).squeeze(-1)


def _sample_from_probs(probs: torch.Tensor, top_p: float) -> torch.Tensor:
    """Sample one token from an already-computed probability vector with optional top-p.

    Args:
        probs: (vocab_size,) — probability distribution (already softmax'd).
        top_p: Nucleus sampling threshold. Use 1.0 to skip.

    Returns:
        Scalar tensor — sampled token id.
    """
    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        sorted_probs[cumulative - sorted_probs > top_p] = 0.0
        s = sorted_probs.sum()
        if s < 1e-10:
            sorted_probs = probs[sorted_indices]  # fallback: restore original order
        else:
            sorted_probs /= s
        probs = torch.zeros_like(probs).scatter_(0, sorted_indices, sorted_probs)

    return torch.multinomial(probs, 1).squeeze(-1)


def _get_probs(model: nn.Module, input_ids: torch.Tensor, temperature: float) -> torch.Tensor:
    """Run model forward and return probability distribution at each position.

    Args:
        input_ids: (1, seq_len)

    Returns:
        (seq_len, vocab_size) — softmax probabilities.
    """
    _, logits, _ = model(input_ids)
    if temperature != 1.0:
        logits = logits / temperature
    return F.softmax(logits[0], dim=-1)  # (seq_len, vocab_size)


class SpeculativeDecoder:
    """Generates tokens using speculative decoding.

    Uses a small draft model to speculatively propose K tokens, then
    verifies them in parallel with a target model using rejection sampling.

    Args:
        draft_model: Small fast model for proposing tokens.
        target_model: Large accurate model for verification.
        cfg: Speculative decoding configuration.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        cfg: SpeculativeConfig | None = None,
    ) -> None:
        self.draft = draft_model
        self.target = target_model
        self.cfg = cfg or SpeculativeConfig()

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Generate tokens with speculative decoding.

        Args:
            input_ids: (1, prompt_len) — input token ids.

        Returns:
            (1, prompt_len + generated_len) — input + generated tokens.
        """
        cfg = self.cfg
        generated = input_ids.clone()
        tokens_generated = 0

        while tokens_generated < cfg.max_new_tokens:
            # Step 1: Draft model proposes K tokens autoregressively
            draft_tokens = []
            draft_probs = []
            draft_input = generated.clone()

            for _ in range(cfg.K):
                d_probs = _get_probs(self.draft, draft_input, cfg.temperature)  # (seq, vocab)
                # d_probs[-1] is already a probability vector — sample directly
                next_token = _sample_from_probs(d_probs[-1], top_p=cfg.top_p)
                draft_tokens.append(next_token)
                draft_probs.append(d_probs[-1])  # prob distribution at last position
                draft_input = torch.cat([draft_input, next_token.view(1, 1)], dim=1)

            # Step 2: Target model verifies all K tokens in one pass
            # Input: original context + all K draft tokens
            verify_input = torch.cat([generated, torch.stack(draft_tokens).unsqueeze(0)], dim=1)
            target_probs_all = _get_probs(self.target, verify_input, cfg.temperature)
            # target_probs_all: (prompt_len + K, vocab_size)
            # target_probs_all[i] is the distribution that predicts token i+1

            # Step 3: Accept/reject each draft token with rejection sampling
            current_len = generated.shape[1]  # snapshot before acceptance modifies generated
            for i, (draft_tok, d_prob) in enumerate(zip(draft_tokens, draft_probs)):
                # target_probs_all at index (current_len - 1 + i) predicts the token at
                # position current_len + i, which is draft_tokens[i]
                target_pos = current_len - 1 + i
                t_prob_tok = target_probs_all[target_pos, draft_tok].clamp(min=1e-10)
                d_prob_tok = d_prob[draft_tok].clamp(min=1e-10)

                accept_prob = torch.clamp(t_prob_tok / d_prob_tok, max=1.0)

                if torch.rand(1, device=generated.device).item() <= accept_prob.item():
                    # Accept: append draft token
                    generated = torch.cat([generated, draft_tok.view(1, 1)], dim=1)
                    tokens_generated += 1

                    if cfg.eos_token_id is not None and draft_tok.item() == cfg.eos_token_id:
                        return generated
                    if tokens_generated >= cfg.max_new_tokens:
                        return generated
                else:
                    # Reject: sample from adjusted distribution max(0, p_target - p_draft)
                    t_full = target_probs_all[target_pos]
                    adjusted = (t_full - d_prob).clamp(min=0.0)
                    if adjusted.sum() < 1e-10:
                        adjusted = t_full.clone()
                    adjusted /= adjusted.sum()
                    corrected_tok = torch.multinomial(adjusted, 1).squeeze(-1)
                    generated = torch.cat([generated, corrected_tok.view(1, 1)], dim=1)
                    tokens_generated += 1

                    if cfg.eos_token_id is not None and corrected_tok.item() == cfg.eos_token_id:
                        return generated
                    break  # stop verifying remaining drafts; restart speculation
            else:
                # All K accepted: sample bonus token from target distribution at position K
                # target_probs_all[current_len + K - 1] predicts the next token after
                # all accepted draft tokens
                bonus_pos = current_len + cfg.K - 1
                bonus_tok = _sample_from_probs(target_probs_all[bonus_pos], top_p=cfg.top_p)
                generated = torch.cat([generated, bonus_tok.view(1, 1)], dim=1)
                tokens_generated += 1

                if cfg.eos_token_id is not None and bonus_tok.item() == cfg.eos_token_id:
                    return generated

        return generated

    def acceptance_rate(
        self,
        input_ids: torch.Tensor,
        n_rounds: int = 10,
    ) -> float:
        """Estimate acceptance rate over n_rounds of speculation.

        Returns average fraction of K draft tokens accepted per round.
        """
        total_accepted = 0
        total_proposed = 0
        cfg = self.cfg
        generated = input_ids.clone()

        for _ in range(n_rounds):
            draft_tokens = []
            draft_probs = []
            draft_input = generated.clone()

            for _ in range(cfg.K):
                d_probs = _get_probs(self.draft, draft_input, cfg.temperature)
                next_token = _sample_from_probs(d_probs[-1], top_p=cfg.top_p)
                draft_tokens.append(next_token)
                draft_probs.append(d_probs[-1])
                draft_input = torch.cat([draft_input, next_token.view(1, 1)], dim=1)

            verify_input = torch.cat([generated, torch.stack(draft_tokens).unsqueeze(0)], dim=1)
            target_probs_all = _get_probs(self.target, verify_input, cfg.temperature)

            round_accepted = 0
            current_len = generated.shape[1]  # snapshot before acceptance modifies generated
            for i, (draft_tok, d_prob) in enumerate(zip(draft_tokens, draft_probs)):
                target_pos = current_len - 1 + i
                t_prob_tok = target_probs_all[target_pos, draft_tok].clamp(min=1e-10)
                d_prob_tok = d_prob[draft_tok].clamp(min=1e-10)
                accept_prob = torch.clamp(t_prob_tok / d_prob_tok, max=1.0)

                if torch.rand(1, device=generated.device).item() <= accept_prob.item():
                    generated = torch.cat([generated, draft_tok.view(1, 1)], dim=1)
                    round_accepted += 1
                else:
                    break

            total_accepted += round_accepted
            total_proposed += cfg.K

            if generated.shape[1] > input_ids.shape[1] + 50:
                break  # enough context

        return total_accepted / max(total_proposed, 1)
