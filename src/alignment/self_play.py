"""Aurelius -- Self-Play Training (SPIN) with win-rate evaluation.

Provides generation, evaluation, and training utilities for self-play
alignment where the model iteratively improves by distinguishing real
human responses from its own synthetic outputs.

Components:
    - SelfPlayConfig: hyperparameters for self-play training
    - generate_self_play_response: temperature sampling generation
    - compute_win_rate: head-to-head evaluation via a judge function
    - spin_loss: SPIN objective loss computation
    - SelfPlayTrainer: orchestrates rounds of self-play training
"""

from __future__ import annotations

import copy
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SelfPlayConfig:
    """Hyperparameters for self-play training.

    Args:
        n_rounds: Number of self-play rounds. Default: 3.
        beta: Temperature coefficient for SPIN loss. Default: 0.1.
        max_gen_tokens: Maximum tokens to generate per response. Default: 64.
        temperature: Sampling temperature for generation. Default: 0.8.
        improvement_threshold: Minimum win-rate improvement to continue. Default: 0.01.
    """

    n_rounds: int = 3
    beta: float = 0.1
    max_gen_tokens: int = 64
    temperature: float = 0.8
    improvement_threshold: float = 0.01


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def generate_self_play_response(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    max_tokens: int,
    temperature: float,
) -> torch.Tensor:
    """Generate a response via temperature sampling.

    Args:
        model: Model whose forward returns (loss, logits, present_key_values).
        prompt_ids: Prompt token ids. Shape (1, S).
        max_tokens: Maximum number of tokens to generate.
        temperature: Sampling temperature (must be > 0).

    Returns:
        Generated token ids tensor of shape (1, max_tokens).
    """
    if temperature <= 0:
        raise ValueError(f"temperature must be > 0, got {temperature}")
    if max_tokens <= 0:
        raise ValueError(f"max_tokens must be > 0, got {max_tokens}")

    model.eval()
    generated = []
    current_ids = prompt_ids

    with torch.no_grad():
        for _ in range(max_tokens):
            _, logits, _ = model(current_ids)
            next_logits = logits[:, -1, :] / temperature
            probs = F.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
            generated.append(next_token)
            current_ids = torch.cat([current_ids, next_token], dim=1)

    return torch.cat(generated, dim=1)  # (1, max_tokens)


# ---------------------------------------------------------------------------
# Win-rate evaluation
# ---------------------------------------------------------------------------


def compute_win_rate(
    model: nn.Module,
    opponent_model: nn.Module,
    prompts: list[torch.Tensor],
    judge_fn,
) -> float:
    """Evaluate model vs opponent by generating responses and judging pairs.

    For each prompt, both models generate a response via greedy decoding.
    The judge_fn(response_a, response_b) returns True if response_a wins.

    Args:
        model: Current policy model.
        opponent_model: Opponent (previous iteration) model.
        prompts: List of prompt tensors, each shape (1, S).
        judge_fn: Callable(Tensor, Tensor) -> bool. Returns True if first arg wins.

    Returns:
        Win fraction (float in [0, 1]).
    """
    if len(prompts) == 0:
        raise ValueError("prompts must be non-empty")

    wins = 0
    total = len(prompts)

    model.eval()
    opponent_model.eval()

    with torch.no_grad():
        for prompt in prompts:
            # Greedy decode from both models (single token for efficiency)
            _, logits_a, _ = model(prompt)
            response_a = logits_a[:, -1, :].argmax(dim=-1, keepdim=True)

            _, logits_b, _ = opponent_model(prompt)
            response_b = logits_b[:, -1, :].argmax(dim=-1, keepdim=True)

            if judge_fn(response_a, response_b):
                wins += 1

    return wins / total


# ---------------------------------------------------------------------------
# SPIN loss
# ---------------------------------------------------------------------------


def spin_loss(
    policy_logps_real: torch.Tensor,
    policy_logps_generated: torch.Tensor,
    opponent_logps_real: torch.Tensor,
    opponent_logps_generated: torch.Tensor,
    beta: float = 0.1,
) -> torch.Tensor:
    """Compute the SPIN objective loss.

    L = -log sigmoid(beta * ((policy_real - opponent_real)
                              - (policy_gen - opponent_gen)))

    Args:
        policy_logps_real: Log probs of real data under policy. Shape (B,).
        policy_logps_generated: Log probs of generated data under policy. Shape (B,).
        opponent_logps_real: Log probs of real data under opponent. Shape (B,).
        opponent_logps_generated: Log probs of generated data under opponent. Shape (B,).
        beta: Temperature coefficient. Default: 0.1.

    Returns:
        Scalar loss tensor.
    """
    real_diff = policy_logps_real - opponent_logps_real
    gen_diff = policy_logps_generated - opponent_logps_generated
    logits = beta * (real_diff - gen_diff)
    return -F.logsigmoid(logits).mean()


# ---------------------------------------------------------------------------
# Token log-probability helper
# ---------------------------------------------------------------------------


def _compute_token_log_probs(
    model: nn.Module,
    input_ids: torch.Tensor,
    response_ids: torch.Tensor,
) -> torch.Tensor:
    """Compute sum of log probabilities for response tokens.

    Args:
        model: Model whose forward returns (loss, logits, present_key_values).
        input_ids: Prompt token ids. Shape (B, S).
        response_ids: Response token ids. Shape (B, R).

    Returns:
        Shape (B,) -- per-sequence log probability of the response.
    """
    B, S = input_ids.shape
    _, R = response_ids.shape

    full_ids = torch.cat([input_ids, response_ids], dim=1)
    _, logits, _ = model(full_ids)

    response_logits = logits[:, S - 1 : S + R - 1, :]
    log_probs = F.log_softmax(response_logits, dim=-1)
    token_log_probs = log_probs.gather(2, response_ids.unsqueeze(-1)).squeeze(-1)

    return token_log_probs.sum(dim=-1)


# ---------------------------------------------------------------------------
# Self-Play Trainer
# ---------------------------------------------------------------------------


class SelfPlayTrainer:
    """Orchestrates self-play training rounds.

    Each round:
        1. Generate synthetic responses from the current model.
        2. Compute log probabilities for real and synthetic responses.
        3. Train to prefer real over synthetic via spin_loss.

    Args:
        model: The model to train.
        config: SelfPlayConfig with hyperparameters.
        optimizer: Optimizer for model parameters.
    """

    def __init__(
        self,
        model: nn.Module,
        config: SelfPlayConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.opponent_model = copy.deepcopy(model)
        for p in self.opponent_model.parameters():
            p.requires_grad_(False)

    def train_round(
        self,
        real_data: list[dict],
        prompts: list[torch.Tensor],
    ) -> dict:
        """Run one round of self-play training.

        Args:
            real_data: List of dicts with 'prompt_ids' (1, S) and
                       'response_ids' (1, R) tensors.
            prompts: List of prompt tensors for win-rate evaluation.

        Returns:
            Dict with 'round_loss' (float) and 'win_rate' (float).
        """
        total_loss = 0.0
        n_steps = 0

        self.model.train()

        for item in real_data:
            prompt_ids = item["prompt_ids"]
            real_response_ids = item["response_ids"]

            # Generate synthetic response from current model snapshot (opponent)
            synth_response_ids = generate_self_play_response(
                self.opponent_model,
                prompt_ids,
                max_tokens=real_response_ids.shape[1],
                temperature=self.config.temperature,
            )

            # Compute policy log probs
            policy_logps_real = _compute_token_log_probs(
                self.model,
                prompt_ids,
                real_response_ids,
            )
            policy_logps_gen = _compute_token_log_probs(
                self.model,
                prompt_ids,
                synth_response_ids,
            )

            # Compute opponent log probs (no grad)
            with torch.no_grad():
                opp_logps_real = _compute_token_log_probs(
                    self.opponent_model,
                    prompt_ids,
                    real_response_ids,
                )
                opp_logps_gen = _compute_token_log_probs(
                    self.opponent_model,
                    prompt_ids,
                    synth_response_ids,
                )

            loss = spin_loss(
                policy_logps_real,
                policy_logps_gen,
                opp_logps_real,
                opp_logps_gen,
                beta=self.config.beta,
            )

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            n_steps += 1

        # Update opponent to current model weights for next round
        self.opponent_model.load_state_dict(copy.deepcopy(self.model.state_dict()))
        for p in self.opponent_model.parameters():
            p.requires_grad_(False)

        avg_loss = total_loss / max(n_steps, 1)

        # Compute win rate using a simple judge
        def _default_judge(resp_a, resp_b):
            return resp_a.sum().item() >= resp_b.sum().item()

        win_rate = (
            compute_win_rate(self.model, self.opponent_model, prompts, _default_judge)
            if prompts
            else 0.0
        )

        return {
            "round_loss": avg_loss,
            "win_rate": win_rate,
        }
