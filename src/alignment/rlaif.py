"""Aurelius — RLAIF (Reinforcement Learning from AI Feedback).

Uses a judge model to score and rank candidate responses, then applies a
DPO-style preference loss to the policy model.  The full pipeline:

    Generate N candidate responses from the policy model
    → Score each with the judge model (log-probability of response)
    → Rank responses by score
    → Extract best/worst as preference pair (chosen, rejected)
    → Compute DPO-like loss and update policy
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RLAIFConfig:
    """Configuration for RLAIF training.

    Args:
        n_samples: Number of candidate responses to generate per prompt.
        beta: DPO temperature / KL penalty coefficient.
        ai_judge_temperature: Temperature used when the judge model scores.
        max_response_tokens: Maximum tokens per generated response.
        reward_scale: Multiplicative scale applied to judge scores.
    """

    n_samples: int = 4
    beta: float = 0.1
    ai_judge_temperature: float = 0.7
    max_response_tokens: int = 64
    reward_scale: float = 1.0


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


@torch.no_grad()
def generate_response(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    max_tokens: int,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Generate a completion from *model* using temperature sampling.

    Args:
        model: Model with forward(input_ids) -> (loss, logits, past_kv).
        prompt_ids: (1, prompt_len) token ids for the prompt.
        max_tokens: Number of new tokens to generate.
        temperature: Sampling temperature (>0).

    Returns:
        (1, max_tokens) tensor of generated token ids.
    """
    generated: list[torch.Tensor] = []
    cur_ids = prompt_ids

    for _ in range(max_tokens):
        _loss, logits, _pkv = model(cur_ids)
        next_logits = logits[:, -1, :]  # (1, vocab)
        if temperature != 1.0:
            next_logits = next_logits / temperature
        probs = next_logits.softmax(dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
        generated.append(next_token)
        cur_ids = next_token

    if not generated:
        return torch.zeros((1, 0), dtype=torch.long, device=prompt_ids.device)

    return torch.cat(generated, dim=1)  # (1, max_tokens)


# ---------------------------------------------------------------------------
# AI Judge scoring
# ---------------------------------------------------------------------------


@torch.no_grad()
def ai_judge_score(
    judge_model: nn.Module,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
) -> float:
    """Score a response using the judge model's log-probability.

    Feeds [prompt || response] through the judge model and computes the
    mean log-probability of the response tokens given the prompt context.

    Args:
        judge_model: Model with forward(input_ids) -> (loss, logits, past_kv).
        prompt_ids: (1, prompt_len) prompt token ids.
        response_ids: (1, response_len) response token ids.

    Returns:
        Mean log-probability of response tokens (float, typically negative).
    """
    full_ids = torch.cat([prompt_ids, response_ids], dim=1)  # (1, total)
    _loss, logits, _pkv = judge_model(full_ids)

    # logits shape: (1, total, vocab)
    # We want the log-prob of each response token given preceding context.
    # Token at position t is predicted by logits at position t-1.
    prompt_len = prompt_ids.shape[1]
    response_len = response_ids.shape[1]

    if response_len == 0:
        return 0.0

    # Logits that predict response tokens: positions [prompt_len-1 .. total-2]
    pred_logits = logits[
        :, prompt_len - 1 : prompt_len + response_len - 1, :
    ]  # (1, response_len, vocab)
    log_probs = F.log_softmax(pred_logits, dim=-1)  # (1, response_len, vocab)

    # Gather log-probs of actual response tokens
    target_ids = response_ids.unsqueeze(-1)  # (1, response_len, 1)
    token_log_probs = log_probs.gather(dim=-1, index=target_ids).squeeze(-1)  # (1, response_len)

    mean_lp = token_log_probs.mean().item()
    return float(mean_lp)


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def rank_responses(
    judge_model: nn.Module,
    prompt_ids: torch.Tensor,
    responses: list[torch.Tensor],
) -> list[tuple[torch.Tensor, float]]:
    """Score all responses with the judge and return sorted (descending) list.

    Args:
        judge_model: Judge model for scoring.
        prompt_ids: (1, prompt_len) prompt token ids.
        responses: List of (1, response_len_i) response tensors.

    Returns:
        List of (response_tensor, score) tuples sorted best-first (highest score).
    """
    scored: list[tuple[torch.Tensor, float]] = []
    for resp in responses:
        score = ai_judge_score(judge_model, prompt_ids, resp)
        scored.append((resp, score))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored


# ---------------------------------------------------------------------------
# Preference pair extraction
# ---------------------------------------------------------------------------


def preference_pair_from_rankings(
    rankings: list[tuple[torch.Tensor, float]],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract the best and worst response from ranked list.

    Args:
        rankings: Sorted list of (response, score) tuples (best first).

    Returns:
        (chosen, rejected) tuple of response tensors.
    """
    chosen = rankings[0][0]
    rejected = rankings[-1][0]
    return chosen, rejected


# ---------------------------------------------------------------------------
# Log-prob computation for policy gradient
# ---------------------------------------------------------------------------


def _sequence_log_probs(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    response_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute per-token log-probs of response under model (with grad).

    Args:
        model: Policy or reference model.
        prompt_ids: (1, prompt_len).
        response_ids: (1, response_len).

    Returns:
        Scalar: sum of log-probs over response tokens.
    """
    full_ids = torch.cat([prompt_ids, response_ids], dim=1)
    _loss, logits, _pkv = model(full_ids)

    prompt_len = prompt_ids.shape[1]
    response_len = response_ids.shape[1]

    pred_logits = logits[:, prompt_len - 1 : prompt_len + response_len - 1, :]
    log_probs = F.log_softmax(pred_logits, dim=-1)

    target_ids = response_ids.unsqueeze(-1)
    token_log_probs = log_probs.gather(dim=-1, index=target_ids).squeeze(-1)

    return token_log_probs.sum()


# ---------------------------------------------------------------------------
# RLAIFTrainer
# ---------------------------------------------------------------------------


class RLAIFTrainer:
    """RLAIF trainer: generate, judge, preference-pair, DPO-style update.

    Args:
        policy_model: The model being trained.
        judge_model: The model used to score/rank responses (can be the same model).
        config: RLAIFConfig with hyperparameters.
        optimizer: Optimizer for the policy model.
    """

    def __init__(
        self,
        policy_model: nn.Module,
        judge_model: nn.Module,
        config: RLAIFConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.policy_model = policy_model
        self.judge_model = judge_model
        self.config = config
        self.optimizer = optimizer

    def train_step(self, prompt_ids: torch.Tensor) -> dict:
        """Run one RLAIF training step.

        1. Generate n_samples responses from policy.
        2. Rank via judge model.
        3. Extract preference pair (chosen, rejected).
        4. Compute DPO-like loss and update policy.

        Args:
            prompt_ids: (1, prompt_len) prompt token ids.

        Returns:
            dict with keys: loss, mean_score, best_score, n_samples.
        """
        self.policy_model.eval()
        self.judge_model.eval()

        # 1. Generate candidate responses
        responses: list[torch.Tensor] = []
        for i in range(self.config.n_samples):
            resp = generate_response(
                self.policy_model,
                prompt_ids,
                max_tokens=self.config.max_response_tokens,
                temperature=self.config.ai_judge_temperature,
            )
            responses.append(resp)

        # 2. Rank via judge
        rankings = rank_responses(self.judge_model, prompt_ids, responses)

        # Collect scores
        scores = [s for _, s in rankings]
        mean_score = sum(scores) / len(scores) if scores else 0.0
        best_score = scores[0] if scores else 0.0

        # Apply reward scale
        mean_score *= self.config.reward_scale
        best_score *= self.config.reward_scale

        # 3. Preference pair
        chosen, rejected = preference_pair_from_rankings(rankings)

        # 4. DPO-like loss: -log sigma(beta * (log pi(chosen) - log pi(rejected)))
        self.policy_model.train()
        self.optimizer.zero_grad()

        chosen_lp = _sequence_log_probs(self.policy_model, prompt_ids, chosen)
        rejected_lp = _sequence_log_probs(self.policy_model, prompt_ids, rejected)

        diff = self.config.beta * (chosen_lp - rejected_lp)
        loss = -F.logsigmoid(diff)

        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "mean_score": mean_score,
            "best_score": best_score,
            "n_samples": self.config.n_samples,
        }
