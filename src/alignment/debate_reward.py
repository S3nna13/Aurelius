"""Debate-based reward modeling for RLHF.

Two agents argue pro/con positions on a question; a judge model scores
each argument using log-probability. The debate outcome drives reward
signal for training a reward model via Bradley-Terry preference loss.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DebateConfig:
    """Configuration for debate-based reward modeling."""

    n_rounds: int = 2
    max_argument_tokens: int = 16
    judge_temperature: float = 0.5
    reward_scale: float = 1.0


class ArgumentGenerator:
    """Generates token-level arguments for a given position.

    Uses the model autoregressively to produce `max_argument_tokens` new
    tokens conditioned on context_ids plus a single position token (the
    first byte of the UTF-8 encoding of the position string).
    """

    def __init__(self, model: nn.Module, config: DebateConfig) -> None:
        self.model = model
        self.config = config

    def generate_argument(
        self,
        context_ids: torch.Tensor,
        position: str,
    ) -> torch.Tensor:
        """Generate an argument for *position*.

        Args:
            context_ids: (1, S) — context token ids.
            position: "pro" or "con" (first byte used as position token).

        Returns:
            (1, max_argument_tokens) — generated token ids.
        """
        # Encode position: use first byte of UTF-8 as a single token id.
        position_token = position.encode("utf-8")[0]  # int in [0, 255]
        pos_tensor = torch.tensor([[position_token]], dtype=torch.long)

        # Prepend position token to context.
        prefix = torch.cat([pos_tensor, context_ids], dim=1)  # (1, S+1)

        generated: list[int] = []
        current_ids = prefix

        with torch.no_grad():
            for _ in range(self.config.max_argument_tokens):
                loss, logits, _pkv = self.model(current_ids)
                next_logits = logits[0, -1, :]  # (vocab_size,)

                if self.config.judge_temperature > 0:
                    next_logits = next_logits / self.config.judge_temperature
                    probs = torch.softmax(next_logits, dim=-1)
                    next_token = int(torch.multinomial(probs, 1).item())
                else:
                    next_token = int(next_logits.argmax().item())

                generated.append(next_token)
                new_tok = torch.tensor([[next_token]], dtype=torch.long)
                current_ids = torch.cat([current_ids, new_tok], dim=1)

        return torch.tensor([generated], dtype=torch.long)  # (1, max_argument_tokens)


class JudgeModel:
    """Scores arguments using the transformer as a judge.

    Quality is measured as the mean log-probability of the argument tokens
    under the model — higher means the model assigns higher likelihood to
    that argument.
    """

    def __init__(self, model: nn.Module, config: DebateConfig) -> None:
        self.model = model
        self.config = config

    def score_argument(self, argument_ids: torch.Tensor) -> float:
        """Score an argument sequence.

        Args:
            argument_ids: (1, T) — argument token ids.

        Returns:
            Mean log-probability of the argument tokens (float, higher = better).
        """
        if argument_ids.shape[1] == 0:
            return 0.0

        with torch.no_grad():
            _loss, logits, _pkv = self.model(argument_ids)
            # logits: (1, T, vocab_size)
            # Predict each token from the previous one (shift by 1).
            # log-probs for positions 1..T-1 predicted from 0..T-2.
            if argument_ids.shape[1] == 1:
                # Single token: score from position 0 logit only.
                log_probs = F.log_softmax(logits[0, 0, :], dim=-1)
                token_id = argument_ids[0, 0].item()
                return float(log_probs[token_id].item())

            shift_logits = logits[0, :-1, :]  # (T-1, vocab_size)
            shift_labels = argument_ids[0, 1:]  # (T-1,)
            log_probs = F.log_softmax(shift_logits, dim=-1)  # (T-1, vocab_size)
            token_log_probs = log_probs[
                torch.arange(shift_labels.shape[0]), shift_labels
            ]  # (T-1,)
            return float(token_log_probs.mean().item())

    def compare(
        self, ids_a: torch.Tensor, ids_b: torch.Tensor
    ) -> tuple[float, float]:
        """Score two arguments and return probabilities that sum to 1.

        Args:
            ids_a: (1, T_a) — first argument token ids.
            ids_b: (1, T_b) — second argument token ids.

        Returns:
            (score_a, score_b) normalized via softmax to sum to 1.
        """
        raw_a = self.score_argument(ids_a)
        raw_b = self.score_argument(ids_b)

        # Softmax normalization over [raw_a, raw_b].
        t = torch.tensor([raw_a, raw_b], dtype=torch.float32)
        probs = torch.softmax(t, dim=0)
        return float(probs[0].item()), float(probs[1].item())


def debate_reward(
    argument_generator: ArgumentGenerator,
    judge: JudgeModel,
    question_ids: torch.Tensor,
    n_rounds: int,
) -> dict:
    """Run n_rounds of pro/con debate and return aggregated reward scores.

    Each round:
    - ArgumentGenerator produces a "pro" and a "con" argument.
    - JudgeModel scores each argument.
    Scores are averaged across rounds to produce final pro/con scores.

    Args:
        argument_generator: ArgumentGenerator instance.
        judge: JudgeModel instance.
        question_ids: (1, S) — question context token ids.
        n_rounds: Number of debate rounds to run.

    Returns:
        {
            "pro_score": float,
            "con_score": float,
            "winner": str,   # "pro" or "con"
            "rounds": int,
        }
    """
    pro_total = 0.0
    con_total = 0.0

    for _ in range(n_rounds):
        pro_ids = argument_generator.generate_argument(question_ids, "pro")
        con_ids = argument_generator.generate_argument(question_ids, "con")

        pro_s, con_s = judge.compare(pro_ids, con_ids)
        pro_total += pro_s
        con_total += con_s

    pro_score = pro_total / n_rounds
    con_score = con_total / n_rounds
    winner = "pro" if pro_score >= con_score else "con"

    return {
        "pro_score": pro_score,
        "con_score": con_score,
        "winner": winner,
        "rounds": n_rounds,
    }


class DebateRewardTrainer:
    """Trains a reward model using debate outcomes (Bradley-Terry loss).

    The judge scores chosen/rejected sequences; we minimise:
        loss = -log(sigmoid(score_chosen - score_rejected))
    so that chosen sequences receive higher scores than rejected ones.
    """

    def __init__(
        self,
        model: nn.Module,
        config: DebateConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.judge = JudgeModel(model, config)

    def _score_differentiable(self, argument_ids: torch.Tensor) -> torch.Tensor:
        """Score argument with gradients enabled for training.

        Returns scalar tensor (mean log-prob).
        """
        if argument_ids.shape[1] == 0:
            return torch.tensor(0.0, requires_grad=True)

        _loss, logits, _pkv = self.model(argument_ids)

        if argument_ids.shape[1] == 1:
            log_probs = F.log_softmax(logits[0, 0, :], dim=-1)
            token_id = argument_ids[0, 0]
            return log_probs[token_id]

        shift_logits = logits[0, :-1, :]  # (T-1, vocab_size)
        shift_labels = argument_ids[0, 1:]  # (T-1,)
        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs[torch.arange(shift_labels.shape[0]), shift_labels]
        return token_log_probs.mean()

    def train_step(
        self,
        chosen_ids: torch.Tensor,
        rejected_ids: torch.Tensor,
    ) -> dict:
        """One training step on a (chosen, rejected) preference pair.

        Args:
            chosen_ids: (1, T) — preferred response token ids.
            rejected_ids: (1, T) — dispreferred response token ids.

        Returns:
            {
                "loss": float,
                "chosen_score": float,
                "rejected_score": float,
            }
        """
        self.model.train()
        self.optimizer.zero_grad()

        score_chosen = self._score_differentiable(chosen_ids)
        score_rejected = self._score_differentiable(rejected_ids)

        # Bradley-Terry ranking loss.
        loss = -F.logsigmoid(score_chosen - score_rejected)
        loss.backward()
        self.optimizer.step()

        return {
            "loss": float(loss.item()),
            "chosen_score": float(score_chosen.detach().item()),
            "rejected_score": float(score_rejected.detach().item()),
        }
