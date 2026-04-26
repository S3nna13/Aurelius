"""RLVR Trainer: DeepSeek-R1 / OpenAI o1-style RL from Verifiable Rewards."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class RLVRConfig:
    lr: float = 1e-5
    kl_coef: float = 0.04
    n_rollouts: int = 4  # samples per prompt
    max_new_tokens: int = 16
    temperature: float = 1.0
    clip_eps: float = 0.2  # PPO clip epsilon
    n_ppo_steps: int = 2  # gradient steps per batch
    format_reward: float = 0.1  # reward for correct format
    correctness_reward: float = 1.0
    max_seq_len: int = 64


# ---------------------------------------------------------------------------
# Verifiable Problem
# ---------------------------------------------------------------------------


@dataclass
class VerifiableProblem:
    """A problem with a verifiable answer."""

    prompt_ids: Tensor  # (T_p,) tokenized prompt
    ground_truth: str  # expected answer string (for verification)
    problem_type: str = "math"  # "math", "code", "format"


# ---------------------------------------------------------------------------
# Verifier type alias
# ---------------------------------------------------------------------------

# Takes (generated_text, ground_truth) -> reward in [0, 1]
VerifierFn = Callable[[str, str], float]


# ---------------------------------------------------------------------------
# Verifier functions
# ---------------------------------------------------------------------------


def math_verifier(generated: str, ground_truth: str) -> float:
    """Extract last number from generated, compare to ground_truth.

    Returns 1.0 if match, 0.0 if not. Handles int/float comparison.
    """
    numbers = re.findall(r"-?\d+\.?\d*", generated)
    if not numbers:
        return 0.0
    try:
        predicted = float(numbers[-1])
        truth = float(ground_truth)
    except ValueError:
        return 0.0
    if abs(predicted - truth) < 1e-6:
        return 1.0
    return 0.0


def format_verifier(generated: str, ground_truth: str) -> float:
    """Check if generated text contains expected format markers.

    ground_truth is a regex pattern or substring to find.
    Returns 1.0 if found, 0.5 if partial (contains some words), 0.0 if none.
    """
    # First try full regex / substring match
    try:
        if re.search(ground_truth, generated):
            return 1.0
    except re.error:
        if ground_truth in generated:
            return 1.0

    # Partial: check individual words of ground_truth
    words = ground_truth.split()
    if not words:
        return 0.0
    matched = sum(1 for w in words if w in generated)
    if matched > 0:
        return 0.5
    return 0.0


# ---------------------------------------------------------------------------
# Generation helpers
# ---------------------------------------------------------------------------


def _generate_single(
    model: nn.Module,
    prompt_ids: Tensor,
    max_new_tokens: int,
    temperature: float,
    max_seq_len: int,
) -> list[int]:
    """Generate up to max_new_tokens tokens with temperature sampling."""
    model.eval()
    with torch.no_grad():
        # prompt_ids shape: (T_p,) — make it (1, T_p)
        if prompt_ids.dim() == 1:
            cur_ids = prompt_ids.unsqueeze(0)
        else:
            cur_ids = prompt_ids.clone()

        generated: list[int] = []
        for _ in range(max_new_tokens):
            # Trim to max_seq_len if needed
            if cur_ids.shape[1] > max_seq_len:
                cur_ids = cur_ids[:, -max_seq_len:]

            out = model(cur_ids)
            # Handle tuple output (loss, logits, pkv) or just logits
            if isinstance(out, tuple):
                logits = out[1]
            else:
                logits = out

            next_logits = logits[:, -1, :]  # (1, V)
            if temperature != 1.0 and temperature > 0.0:
                next_logits = next_logits / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
            token_id = int(next_token.item())
            generated.append(token_id)
            cur_ids = torch.cat([cur_ids, next_token], dim=1)

    return generated


# ---------------------------------------------------------------------------
# generate_rollouts
# ---------------------------------------------------------------------------


def generate_rollouts(
    model: nn.Module,
    problems: list[VerifiableProblem],
    config: RLVRConfig,
    tokenizer_decode: Callable[[list[int]], str],
    verifier: VerifierFn,
) -> list[dict]:
    """For each problem, generate n_rollouts responses and compute verifiable rewards.

    Returns list of dicts:
        {'prompt_ids', 'response_ids', 'reward', 'decoded_response'}
    """
    rollouts: list[dict] = []
    for problem in problems:
        for _ in range(config.n_rollouts):
            response_ids = _generate_single(
                model,
                problem.prompt_ids,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature,
                max_seq_len=config.max_seq_len,
            )
            decoded = tokenizer_decode(response_ids)
            reward = verifier(decoded, problem.ground_truth)
            rollouts.append(
                {
                    "prompt_ids": problem.prompt_ids,
                    "response_ids": response_ids,
                    "reward": float(reward),
                    "decoded_response": decoded,
                }
            )
    return rollouts


# ---------------------------------------------------------------------------
# Log-prob recomputation
# ---------------------------------------------------------------------------


def _recompute_log_probs(
    model: nn.Module,
    prompt_ids: Tensor,
    response_ids: list[int],
) -> Tensor:
    """Recompute per-token log probs for a response given a prompt. Returns (T_r,)."""
    resp_tensor = torch.tensor(response_ids, dtype=torch.long)
    # prompt_ids: (T_p,) or (1, T_p)
    if prompt_ids.dim() == 1:
        p = prompt_ids.unsqueeze(0)  # (1, T_p)
    else:
        p = prompt_ids

    r = resp_tensor.unsqueeze(0)  # (1, T_r)
    full_ids = torch.cat([p, r], dim=1)  # (1, T_p + T_r)

    out = model(full_ids)
    if isinstance(out, tuple):
        logits = out[1]
    else:
        logits = out

    log_probs_all = F.log_softmax(logits[:, :-1, :], dim=-1)  # (1, T_p+T_r-1, V)
    T_p = p.shape[1]
    T_r = len(response_ids)
    # response positions start at T_p - 1 in the shifted logits
    comp_lp = log_probs_all[:, T_p - 1 : T_p - 1 + T_r, :]  # (1, T_r, V)
    token_lp = comp_lp.squeeze(0).gather(1, resp_tensor.unsqueeze(1)).squeeze(1)  # (T_r,)
    return token_lp


# ---------------------------------------------------------------------------
# RLVR loss
# ---------------------------------------------------------------------------


def rlvr_loss(
    model: nn.Module,
    ref_model: nn.Module,
    rollouts: list[dict],
    config: RLVRConfig,
) -> tuple[Tensor, dict[str, float]]:
    """PPO-style loss with KL penalty.

    L = -E[min(ratio * A, clip(ratio, 1+/-eps) * A)] + kl_coef * KL(policy || ref)

    where A = normalized(rewards - mean_reward) across rollouts.

    Returns (loss, {'policy_loss', 'kl', 'mean_reward', 'reward_std'})
    """
    rewards = torch.tensor([r["reward"] for r in rollouts], dtype=torch.float32)
    mean_reward = rewards.mean()
    std_reward = rewards.std() + 1e-8
    advantages = (rewards - mean_reward) / std_reward  # (N,)

    policy_losses: list[Tensor] = []
    kl_terms: list[Tensor] = []

    for i, rollout in enumerate(rollouts):
        prompt_ids = rollout["prompt_ids"]
        response_ids = rollout["response_ids"]

        if len(response_ids) == 0:
            continue

        # Policy log probs (with gradient)
        policy_lp = _recompute_log_probs(model, prompt_ids, response_ids)  # (T_r,)
        policy_lp_sum = policy_lp.sum()

        # Reference log probs (no gradient)
        with torch.no_grad():
            ref_lp = _recompute_log_probs(ref_model, prompt_ids, response_ids)
            ref_lp_sum = ref_lp.sum()

        ratio = torch.exp(policy_lp_sum - ref_lp_sum.detach())
        adv = advantages[i]

        surr1 = ratio * adv
        surr2 = torch.clamp(ratio, 1.0 - config.clip_eps, 1.0 + config.clip_eps) * adv
        policy_losses.append(-torch.min(surr1, surr2))

        # KL: mean over tokens of (policy_lp - ref_lp)
        kl_per_token = policy_lp - ref_lp.detach()
        kl_terms.append(kl_per_token.mean())

    if not policy_losses:
        zero = torch.tensor(0.0, requires_grad=True)
        metrics: dict[str, float] = {
            "policy_loss": 0.0,
            "kl": 0.0,
            "mean_reward": float(mean_reward.item()),
            "reward_std": float(std_reward.item()),
        }
        return zero, metrics

    policy_loss = torch.stack(policy_losses).mean()
    kl = torch.stack(kl_terms).mean()

    loss = policy_loss + config.kl_coef * kl

    metrics = {
        "policy_loss": float(policy_loss.item()),
        "kl": float(kl.item()),
        "mean_reward": float(mean_reward.item()),
        "reward_std": float(std_reward.item()),
    }
    return loss, metrics


# ---------------------------------------------------------------------------
# RLVRTrainer
# ---------------------------------------------------------------------------


class RLVRTrainer:
    """Training orchestrator for RLVR (DeepSeek-R1 / o1 style)."""

    def __init__(
        self,
        model: nn.Module,
        ref_model: nn.Module,
        config: RLVRConfig,
        verifier: VerifierFn,
        tokenizer_decode: Callable[[list[int]], str] | None = None,
    ):
        self.model = model
        self.ref_model = ref_model
        self.config = config
        self.verifier = verifier

        if tokenizer_decode is None:
            self.tokenizer_decode: Callable[[list[int]], str] = lambda ids: bytes(
                [i % 256 for i in ids]
            ).decode("utf-8", errors="replace")
        else:
            self.tokenizer_decode = tokenizer_decode

        self.optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr)

        # Freeze ref model
        for p in self.ref_model.parameters():
            p.requires_grad_(False)
        self.ref_model.eval()

    def step(self, problems: list[VerifiableProblem]) -> dict[str, float]:
        """Full RLVR step: generate rollouts, compute rewards, PPO update.

        Returns metrics dict.
        """
        # 1. Generate rollouts (no grad)
        rollouts = generate_rollouts(
            self.model,
            problems,
            self.config,
            self.tokenizer_decode,
            self.verifier,
        )

        # 2. PPO gradient steps
        last_metrics: dict[str, float] = {}
        self.model.train()
        for _ in range(self.config.n_ppo_steps):
            self.optimizer.zero_grad()
            loss, metrics = rlvr_loss(
                self.model,
                self.ref_model,
                rollouts,
                self.config,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            last_metrics = metrics
            last_metrics["loss"] = float(loss.item())

        return last_metrics
