"""Speculative rejection sampling: token-level quality filtering during generation.

At each step, a lightweight quality scorer evaluates the partial sequence.
Tokens are rejected (and resampled) if the quality score drops below threshold.

This maintains the target distribution approximately while filtering toxic/
low-quality prefixes before they can cascade into full bad responses.

Reference: arXiv:2410.07524 "Speculative Rejection: Improving Language Model
Efficiency with Token-level Quality Assessment"
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn.functional as F


@dataclass
class SpeculativeRejectionConfig:
    max_new_tokens: int = 256
    temperature: float = 1.0
    top_p: float = 0.9
    eos_token_id: int | None = None
    quality_threshold: float = 0.3     # reject if quality score below this
    max_rejections_per_step: int = 5   # max resamples before accepting anyway
    use_log_prob_quality: bool = True  # if True, use log-prob as quality proxy


@dataclass
class RejectionStats:
    """Statistics about rejections during generation."""
    total_steps: int = 0
    total_rejections: int = 0
    tokens_generated: int = 0

    @property
    def rejection_rate(self) -> float:
        if self.total_steps == 0:
            return 0.0
        return self.total_rejections / self.total_steps


def log_prob_quality_score(
    logits: torch.Tensor,      # (vocab_size,) logits for current step
    sampled_token: int,        # the token sampled
    temperature: float = 1.0,
) -> float:
    """Quality score based on token log-probability.

    Returns the log-probability of the sampled token under the distribution.
    Higher = the model was more confident = higher quality token.
    Normalized to [0, 1] using sigmoid.
    """
    if temperature != 1.0:
        logits = logits / temperature
    log_probs = F.log_softmax(logits, dim=-1)
    log_p = log_probs[sampled_token].item()
    # sigmoid maps (-inf, 0] -> (0, 0.5] for low-prob tokens,
    # and approaches 1 as log_p approaches 0 (i.e., p -> 1).
    return float(torch.sigmoid(torch.tensor(log_p)).item())


def nucleus_sample_with_logit(
    logits: torch.Tensor,   # (vocab_size,)
    top_p: float,
    temperature: float,
) -> tuple[int, float]:
    """Top-p nucleus sampling. Returns (token_id, log_prob_of_token)."""
    if temperature != 1.0:
        logits = logits / temperature

    log_probs = F.log_softmax(logits, dim=-1)
    probs = log_probs.exp()

    if top_p < 1.0:
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative = sorted_probs.cumsum(dim=-1)
        # Keep tokens whose cumulative probability (starting from the top) is
        # within the nucleus.  The mask removes tokens once the cumsum already
        # exceeded top_p *before* adding this token.
        mask = (cumulative - sorted_probs) >= top_p
        sorted_probs = sorted_probs.masked_fill(mask, 0.0)
        s = sorted_probs.sum()
        if s < 1e-10:
            # Fallback: no probability mass survived, reset to full probs
            sorted_probs = probs[sorted_indices]
            s = sorted_probs.sum()
        sorted_probs = sorted_probs / s
        probs = torch.zeros_like(probs).scatter_(0, sorted_indices, sorted_probs)

    token_id = torch.multinomial(probs, 1).squeeze(-1).item()
    log_p = log_probs[token_id].item()
    return int(token_id), float(log_p)


@torch.no_grad()
def speculative_rejection_generate(
    model: torch.nn.Module,
    input_ids: torch.Tensor,      # (1, S) or (S,)
    cfg: SpeculativeRejectionConfig,
    quality_fn=None,               # optional: callable(partial_seq, token, logits) -> float
                                   # if None, uses log_prob_quality_score
) -> tuple[torch.Tensor, RejectionStats]:
    """Generate tokens with speculative rejection sampling.

    For each step:
    1. Get logits from model
    2. Sample token with nucleus sampling
    3. Compute quality score
    4. If quality < threshold AND rejections < max_rejections: reject and resample
    5. Else: accept token (even if below threshold, to avoid infinite loops)

    Returns:
        (generated_ids: (S_new,), stats: RejectionStats)
    """
    # Normalise input shape to (1, S)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    generated = input_ids.clone()
    stats = RejectionStats()

    if quality_fn is None:
        def quality_fn(partial_seq: torch.Tensor, token: int, logits: torch.Tensor) -> float:
            return log_prob_quality_score(logits, token, cfg.temperature)

    generated_tokens: list[int] = []

    for _ in range(cfg.max_new_tokens):
        # Forward pass — model returns (loss, logits, aux) or similar;
        # pattern matches speculative.py which unpacks three values.
        _, logits, _ = model(generated)
        # logits: (1, seq_len, vocab_size) — we want the last position
        step_logits = logits[0, -1, :]  # (vocab_size,)

        stats.total_steps += 1
        rejections_this_step = 0
        accepted_token: int | None = None

        while True:
            token_id, _log_p = nucleus_sample_with_logit(
                step_logits, top_p=cfg.top_p, temperature=cfg.temperature
            )
            score = quality_fn(generated, token_id, step_logits)

            if score >= cfg.quality_threshold:
                # Accept
                accepted_token = token_id
                break
            else:
                # Reject unless we've hit the limit
                if rejections_this_step < cfg.max_rejections_per_step:
                    rejections_this_step += 1
                    stats.total_rejections += 1
                else:
                    # Accept anyway to avoid infinite loop
                    accepted_token = token_id
                    break

        assert accepted_token is not None
        generated_tokens.append(accepted_token)
        stats.tokens_generated += 1

        next_tok = torch.tensor([[accepted_token]], dtype=torch.long, device=generated.device)
        generated = torch.cat([generated, next_tok], dim=1)

        if cfg.eos_token_id is not None and accepted_token == cfg.eos_token_id:
            break

    return torch.tensor(generated_tokens, dtype=torch.long), stats
