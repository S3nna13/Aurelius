"""Speculative decoding v2: draft model + target verification with acceptance rate tracking."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass


@dataclass
class SpeculativeConfig:
    """Configuration for speculative decoding v2."""
    n_draft_tokens: int = 4   # tokens to draft per step
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_k: int = 0            # 0 = disabled
    top_p: float = 1.0


def apply_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    """Set all logits below the k-th largest to -inf.

    Args:
        logits: (..., vocab_size) raw logits.
        k: Number of top logits to keep.

    Returns:
        Modified logits with same shape; non-top-k positions set to -inf.
    """
    if k <= 0:
        return logits
    # Find the k-th largest value along last dim
    top_k_values, _ = torch.topk(logits, k, dim=-1)
    # Threshold: minimum of the top-k values
    threshold = top_k_values[..., -1, None]
    return logits.masked_fill(logits < threshold, float("-inf"))


def apply_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    """Nucleus sampling: mask tokens where cumulative softmax prob > p.

    Keeps at least 1 token.

    Args:
        logits: (..., vocab_size) raw logits.
        p: Nucleus threshold in (0, 1].

    Returns:
        Modified logits with same shape.
    """
    if p >= 1.0:
        return logits

    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumsum = sorted_probs.cumsum(dim=-1)

    # Remove tokens where cumulative prob (shifted by one) > p
    # This ensures we keep all tokens needed to reach probability mass p.
    # Shift right so the token that pushes cumsum over p is still kept.
    remove_mask = (cumsum - sorted_probs) > p
    sorted_probs_filtered = sorted_probs.masked_fill(remove_mask, 0.0)

    # At least one token must survive
    sorted_probs_filtered[..., 0] = sorted_probs[..., 0]

    # Scatter mask back to original order
    mask = torch.zeros_like(logits, dtype=torch.bool)
    mask.scatter_(-1, sorted_indices, remove_mask)
    return logits.masked_fill(mask, float("-inf"))


def sample_token(
    logits: torch.Tensor,
    temperature: float,
    top_k: int,
    top_p: float,
) -> torch.Tensor:
    """Sample one token per batch element from logits.

    Applies temperature scaling, then top-k filtering (if top_k > 0),
    then top-p filtering (if top_p < 1.0), then multinomial sampling.

    Args:
        logits: (B, vocab_size) raw logits for the last position.
        temperature: Sampling temperature.
        top_k: If > 0, restrict to top-k tokens.
        top_p: If < 1.0, apply nucleus sampling.

    Returns:
        (B,) sampled token ids.
    """
    if temperature != 1.0 and temperature > 0:
        logits = logits / temperature

    if top_k > 0:
        logits = apply_top_k(logits, top_k)

    if top_p < 1.0:
        logits = apply_top_p(logits, top_p)

    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1).squeeze(-1)


@torch.no_grad()
def draft_tokens(
    draft_model: nn.Module,
    input_ids: torch.Tensor,
    config: SpeculativeConfig,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Autoregressively generate n_draft_tokens from the draft model.

    Args:
        draft_model: Small fast model.
        input_ids: (B, seq_len) current context.
        config: Speculative decoding configuration.

    Returns:
        (draft_ids, draft_log_probs) both shape (B, n_draft_tokens).
        draft_log_probs[b, t] is the log prob of draft_ids[b, t] under the
        draft distribution at step t.
    """
    B = input_ids.shape[0]
    n = config.n_draft_tokens
    device = input_ids.device
    dtype = input_ids.dtype

    all_draft_ids: list[torch.Tensor] = []
    all_log_probs: list[torch.Tensor] = []

    current = input_ids.clone()

    for _ in range(n):
        _, logits, _ = draft_model(current)   # (B, seq_len_so_far, vocab)
        step_logits = logits[:, -1, :]        # (B, vocab)

        token = sample_token(
            step_logits,
            config.temperature,
            config.top_k,
            config.top_p,
        )  # (B,)

        # Compute log prob of sampled token
        log_prob_dist = F.log_softmax(step_logits, dim=-1)  # (B, vocab)
        log_prob = log_prob_dist.gather(1, token.unsqueeze(1)).squeeze(1)  # (B,)

        all_draft_ids.append(token)
        all_log_probs.append(log_prob)

        current = torch.cat([current, token.unsqueeze(1)], dim=1)

    draft_ids = torch.stack(all_draft_ids, dim=1)      # (B, n)
    draft_log_probs = torch.stack(all_log_probs, dim=1)  # (B, n)
    return draft_ids, draft_log_probs


@torch.no_grad()
def verify_tokens(
    target_model: nn.Module,
    input_ids: torch.Tensor,
    draft_ids: torch.Tensor,
    draft_log_probs: torch.Tensor,
    config: SpeculativeConfig,
) -> tuple[torch.Tensor, int]:
    """Verify draft tokens with the target model via rejection sampling.

    Runs target_model on input_ids concatenated with draft_ids in a single
    forward pass. For each draft token t, accepts with probability
    min(1, p_target[t] / p_draft[t]). On rejection, samples a correction
    token from the adjusted distribution max(0, p_target - p_draft) / norm.

    Args:
        target_model: Large verification model.
        input_ids: (B, seq_len) current context.
        draft_ids: (B, n_draft) draft token ids.
        draft_log_probs: (B, n_draft) log probs of draft tokens under draft model.
        config: Speculative decoding configuration.

    Returns:
        (accepted_ids, n_accepted) where accepted_ids has shape
        (B, accepted_len) and n_accepted is the number of draft tokens
        accepted (scalar, does not count any correction token).
    """
    B, seq_len = input_ids.shape
    n_draft = draft_ids.shape[1]
    device = input_ids.device

    # Single forward pass over context + all draft tokens
    verify_input = torch.cat([input_ids, draft_ids], dim=1)  # (B, seq_len + n_draft)
    _, target_logits, _ = target_model(verify_input)          # (B, seq_len+n_draft, vocab)

    accepted_tokens: list[torch.Tensor] = []
    n_accepted = 0

    for t in range(n_draft):
        # Target distribution for predicting token at position seq_len + t
        # corresponds to target_logits at position seq_len - 1 + t
        target_pos = seq_len - 1 + t
        t_logits = target_logits[:, target_pos, :]  # (B, vocab)

        if config.temperature != 1.0 and config.temperature > 0:
            t_logits = t_logits / config.temperature

        t_log_probs = F.log_softmax(t_logits, dim=-1)  # (B, vocab)
        t_probs = t_log_probs.exp()                    # (B, vocab)

        # Draft probabilities (per-sample, per-token)
        d_log_probs_t = draft_log_probs[:, t]  # (B,)
        d_probs_t = d_log_probs_t.exp().clamp(min=1e-10)  # (B,)

        draft_tok_t = draft_ids[:, t]  # (B,)

        # Acceptance prob for each sample in batch
        t_prob_tok = t_probs.gather(1, draft_tok_t.unsqueeze(1)).squeeze(1).clamp(min=1e-10)  # (B,)
        accept_prob = (t_prob_tok / d_probs_t).clamp(max=1.0)  # (B,)

        # For simplicity with batch size > 1, check first sample for accept/reject
        # (full batched rejection sampling would require per-sample branching)
        # For B=1 this is exact; for B>1 we use batch[0]
        u = torch.rand(B, device=device)
        accepted_mask = u <= accept_prob  # (B,) bool

        if accepted_mask.all():
            # All batch elements accepted this draft token
            accepted_tokens.append(draft_tok_t)
            n_accepted += 1
        else:
            # At least one rejection — use correction token
            # Sample correction from adjusted distribution: max(0, p_target - p_draft)
            d_probs_full = torch.zeros_like(t_probs)
            d_probs_full.scatter_(1, draft_tok_t.unsqueeze(1), d_probs_t.unsqueeze(1))

            adjusted = (t_probs - d_probs_full).clamp(min=0.0)
            row_sums = adjusted.sum(dim=-1, keepdim=True)
            too_small = row_sums < 1e-10
            adjusted = torch.where(too_small, t_probs, adjusted / row_sums.clamp(min=1e-10))

            correction = torch.multinomial(adjusted, num_samples=1).squeeze(1)  # (B,)

            # Use accepted draft token for accepted samples, correction for rejected
            final_tok = torch.where(accepted_mask, draft_tok_t, correction)
            accepted_tokens.append(final_tok)
            break

    if not accepted_tokens:
        # Edge case: nothing accepted (shouldn't normally happen)
        accepted_ids = torch.zeros(B, 0, dtype=input_ids.dtype, device=device)
    else:
        accepted_ids = torch.stack(accepted_tokens, dim=1)  # (B, accepted_len)

    return accepted_ids, n_accepted


class AcceptanceTracker:
    """Tracks cumulative acceptance rate across speculative decoding steps."""

    def __init__(self) -> None:
        self.total_drafted: int = 0
        self.total_accepted: int = 0

    def update(self, n_drafted: int, n_accepted: int) -> None:
        """Record results from one speculative step.

        Args:
            n_drafted: Number of draft tokens proposed.
            n_accepted: Number of draft tokens accepted.
        """
        self.total_drafted += n_drafted
        self.total_accepted += n_accepted

    def acceptance_rate(self) -> float:
        """Return total_accepted / total_drafted (0.0 if no drafts)."""
        if self.total_drafted == 0:
            return 0.0
        return self.total_accepted / self.total_drafted

    def reset(self) -> None:
        """Reset all counters to zero."""
        self.total_drafted = 0
        self.total_accepted = 0


class SpeculativeDecoderV2:
    """Speculative decoder with acceptance rate tracking and speculative sampling."""

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        config: SpeculativeConfig,
    ) -> None:
        self.draft_model = draft_model
        self.target_model = target_model
        self.config = config
        self.tracker = AcceptanceTracker()

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """Generate tokens using speculative decoding.

        Loops: draft n_draft_tokens → verify with target → append accepted
        tokens → repeat until max_new_tokens reached.

        Args:
            input_ids: (B, prompt_len) input token ids.

        Returns:
            (all_generated_ids, stats) where all_generated_ids has shape
            (B, generated_len) and stats contains "acceptance_rate",
            "n_steps", and "total_tokens".
        """
        cfg = self.config
        self.tracker.reset()

        current_ids = input_ids.clone()
        generated_chunks: list[torch.Tensor] = []
        total_tokens = 0
        n_steps = 0

        while total_tokens < cfg.max_new_tokens:
            draft_ids, draft_log_probs = draft_tokens(self.draft_model, current_ids, cfg)
            accepted_ids, n_accepted = verify_tokens(
                self.target_model, current_ids, draft_ids, draft_log_probs, cfg
            )

            self.tracker.update(cfg.n_draft_tokens, n_accepted)

            # Trim to budget
            remaining = cfg.max_new_tokens - total_tokens
            if accepted_ids.shape[1] > remaining:
                accepted_ids = accepted_ids[:, :remaining]

            generated_chunks.append(accepted_ids)
            total_tokens += accepted_ids.shape[1]
            n_steps += 1

            current_ids = torch.cat([current_ids, accepted_ids], dim=1)

            if total_tokens >= cfg.max_new_tokens:
                break

        if generated_chunks:
            all_generated = torch.cat(generated_chunks, dim=1)  # (B, total_tokens)
        else:
            all_generated = torch.zeros(
                input_ids.shape[0], 0, dtype=input_ids.dtype, device=input_ids.device
            )

        stats = {
            "acceptance_rate": self.tracker.acceptance_rate(),
            "n_steps": n_steps,
            "total_tokens": total_tokens,
        }
        return all_generated, stats
