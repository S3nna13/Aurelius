"""Self-play data generation for DPO training.

Generates multiple candidate responses per prompt, scores them by mean
log-probability, and produces (chosen, rejected) pairs suitable for
Direct Preference Optimization.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

import torch
import torch.nn.functional as F


@dataclass
class SelfPlayConfig:
    """Configuration for self-play candidate generation."""

    n_candidates: int = 4
    max_new_tokens: int = 64
    temperature: float = 1.0
    top_p: float = 0.9
    keep_top_k: int = 1
    keep_bottom_k: int = 1


@dataclass
class SelfPlayRollout:
    """A single generated response with its score."""

    prompt_ids: torch.Tensor  # (S_prompt,) 1-D
    response_ids: torch.Tensor  # (S_resp,) 1-D
    score: float  # mean log-prob of response tokens


@dataclass
class SelfPlayPair:
    """A (chosen, rejected) pair for DPO training."""

    prompt_ids: torch.Tensor
    chosen_ids: torch.Tensor
    rejected_ids: torch.Tensor
    chosen_score: float
    rejected_score: float


@torch.no_grad()
def score_response(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
) -> float:
    """Score a response by its mean log-probability under the model.

    Args:
        model: AureliusTransformer instance.
        prompt_ids: 1-D tensor of prompt token ids (S_prompt,).
        response_ids: 1-D tensor of response token ids (S_resp,).

    Returns:
        Mean log-probability of the response tokens (float, <= 0).
    """
    device = next(model.parameters()).device
    full_ids = torch.cat([prompt_ids, response_ids]).unsqueeze(0).to(device)  # (1, S_total)
    S_prompt = prompt_ids.shape[0]
    S_resp = response_ids.shape[0]

    _, logits, _ = model(full_ids)  # (1, S_total, V)

    # Positions [S_prompt-1 .. S_prompt+S_resp-2] predict response tokens
    # i.e. logits at position t predict token at position t+1
    scoring_logits = logits[0, S_prompt - 1 : S_prompt + S_resp - 1, :]  # (S_resp, V)
    log_probs = F.log_softmax(scoring_logits, dim=-1)  # (S_resp, V)

    target = response_ids.to(device)  # (S_resp,)
    token_log_probs = log_probs.gather(1, target.unsqueeze(1)).squeeze(1)  # (S_resp,)

    return token_log_probs.mean().item()


@torch.no_grad()
def generate_candidates(
    model: torch.nn.Module,
    prompt_ids: torch.Tensor,
    cfg: SelfPlayConfig,
) -> list[SelfPlayRollout]:
    """Generate multiple candidate responses and score them.

    Args:
        model: AureliusTransformer instance (must have a ``generate`` method).
        prompt_ids: 1-D tensor of prompt token ids (S_prompt,).
        cfg: Self-play configuration.

    Returns:
        List of :class:`SelfPlayRollout` sorted descending by score.
    """
    device = next(model.parameters()).device
    S_prompt = prompt_ids.shape[0]

    # Batch the prompt n_candidates times -> (n_candidates, S_prompt)
    batch_prompt = prompt_ids.unsqueeze(0).expand(cfg.n_candidates, -1).to(device)

    # Generate -> (n_candidates, S_prompt + max_new_tokens)
    generated = model.generate(
        batch_prompt,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
    )

    # Extract response portion
    responses = generated[:, S_prompt:]  # (n_candidates, new_tokens)

    rollouts: list[SelfPlayRollout] = []
    for i in range(cfg.n_candidates):
        resp = responses[i]  # (new_tokens,)
        score = score_response(model, prompt_ids, resp)
        rollouts.append(
            SelfPlayRollout(
                prompt_ids=prompt_ids.clone(),
                response_ids=resp.cpu(),
                score=score,
            )
        )

    rollouts.sort(key=lambda r: r.score, reverse=True)
    return rollouts


@torch.no_grad()
def make_dpo_pairs(
    model: torch.nn.Module,
    prompts: list[torch.Tensor],
    cfg: SelfPlayConfig,
) -> list[SelfPlayPair]:
    """Generate DPO (chosen, rejected) pairs via self-play.

    For each prompt, generates ``n_candidates`` responses, keeps the top-k
    as chosen and bottom-k as rejected, and forms all cross-product pairs.

    Args:
        model: AureliusTransformer instance.
        prompts: List of 1-D prompt tensors.
        cfg: Self-play configuration.

    Returns:
        List of :class:`SelfPlayPair`.
    """
    pairs: list[SelfPlayPair] = []

    for prompt_ids in prompts:
        rollouts = generate_candidates(model, prompt_ids, cfg)

        # Skip if all scores are effectively equal
        scores = [r.score for r in rollouts]
        if max(scores) - min(scores) < 1e-6:
            continue

        chosen = rollouts[: cfg.keep_top_k]
        rejected = rollouts[-cfg.keep_bottom_k :]

        for c, r in product(chosen, rejected):
            pairs.append(
                SelfPlayPair(
                    prompt_ids=prompt_ids.clone(),
                    chosen_ids=c.response_ids.clone(),
                    rejected_ids=r.response_ids.clone(),
                    chosen_score=c.score,
                    rejected_score=r.score,
                )
            )

    return pairs
