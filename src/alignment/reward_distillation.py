"""Reward model distillation and synthetic preference data generation.

Distills a teacher reward model into a smaller student model using soft labels,
and supports synthetic preference pair generation via NLL-based proxy rewards.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class RewardDistillConfig:
    """Configuration for reward distillation training."""

    temperature: float = 2.0
    alpha: float = 0.5
    n_synthetic_pairs: int = 4
    margin: float = 0.5


class RewardHead(nn.Module):
    """Scalar reward head on top of a transformer's hidden states.

    Takes (B, T, d_model) hidden states, averages over the sequence length,
    and projects to a scalar reward per example.
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, 1, bias=True)
        nn.init.normal_(self.proj.weight, std=0.02)
        nn.init.zeros_(self.proj.bias)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Compute scalar rewards from hidden states.

        Args:
            hidden_states: (B, T, d_model) transformer hidden states.

        Returns:
            (B,) scalar reward per example.
        """
        pooled = hidden_states.mean(dim=1)  # (B, d_model)
        return self.proj(pooled).squeeze(-1)  # (B,)


def distillation_loss(
    student_rewards: torch.Tensor,
    teacher_rewards: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Soft label distillation loss between student and teacher reward distributions.

    Normalizes both reward vectors via softmax at the given temperature,
    then computes MSE between the resulting distributions.

    Args:
        student_rewards: (B,) student reward scores.
        teacher_rewards: (B,) teacher reward scores (treated as targets).
        temperature: Softmax temperature for softening distributions.

    Returns:
        Scalar MSE loss.
    """
    student_soft = F.softmax(student_rewards / temperature, dim=0)
    teacher_soft = F.softmax(teacher_rewards / temperature, dim=0)
    return F.mse_loss(student_soft, teacher_soft)


def preference_loss(
    chosen_rewards: torch.Tensor,
    rejected_rewards: torch.Tensor,
    margin: float,
) -> torch.Tensor:
    """Bradley-Terry preference loss with margin.

    Encourages chosen_rewards > rejected_rewards + margin.

    Args:
        chosen_rewards: (B,) reward scores for preferred responses.
        rejected_rewards: (B,) reward scores for dispreferred responses.
        margin: Minimum required margin between chosen and rejected.

    Returns:
        Scalar loss value.
    """
    return -torch.sigmoid(chosen_rewards - rejected_rewards - margin).log().mean()


def generate_synthetic_pairs(
    model: nn.Module,
    prompt_ids: torch.Tensor,
    n_pairs: int,
    temperature: float,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Generate synthetic (chosen, rejected) pairs from a model.

    Samples two continuations for each pair and uses negative log-likelihood
    as a proxy reward — lower NLL means better quality (chosen).

    Args:
        model: AureliusTransformer. Called as model(input_ids) -> (loss, logits, pkv).
        prompt_ids: (1, S) prompt token ids.
        n_pairs: Number of (chosen, rejected) pairs to generate.
        temperature: Sampling temperature. Use 0 for argmax (greedy).

    Returns:
        List of (chosen_ids, rejected_ids) tensor tuples.
    """
    max_new_tokens = 3
    pairs = []

    model.eval()
    with torch.no_grad():
        for _ in range(n_pairs):
            seqs = []
            nlls = []

            # Generate two candidate sequences
            for _ in range(2):
                current_ids = prompt_ids.clone()
                step_log_probs = []

                for _step in range(max_new_tokens):
                    _loss, logits, _pkv = model(current_ids)
                    last_logits = logits[:, -1, :]  # (1, vocab_size)

                    if temperature == 0:
                        next_token = last_logits.argmax(dim=-1, keepdim=True)  # (1, 1)
                    else:
                        probs = F.softmax(last_logits / temperature, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)

                    log_probs = F.log_softmax(last_logits, dim=-1)
                    token_log_prob = log_probs.gather(1, next_token).squeeze()
                    step_log_probs.append(token_log_prob)

                    current_ids = torch.cat([current_ids, next_token], dim=1)

                nll = -torch.stack(step_log_probs).mean()
                seqs.append(current_ids.squeeze(0))
                nlls.append(nll.item())

            # Lower NLL = chosen
            if nlls[0] <= nlls[1]:
                chosen, rejected = seqs[0], seqs[1]
            else:
                chosen, rejected = seqs[1], seqs[0]

            pairs.append((chosen, rejected))

    return pairs


class RewardDistillationTrainer:
    """Distills a teacher reward model into a student model.

    The teacher is frozen during training. Combines distillation loss
    (soft label matching) and preference loss (Bradley-Terry with margin).

    Args:
        student_model: AureliusTransformer to train as student reward model.
        teacher_model: Frozen AureliusTransformer used as teacher.
        config: Distillation configuration.
        optimizer: Optimizer for the student model parameters.
    """

    def __init__(
        self,
        student_model: nn.Module,
        teacher_model: nn.Module,
        config: RewardDistillConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.student_model = student_model
        self.teacher_model = teacher_model
        self.config = config
        self.optimizer = optimizer

        d_model = student_model.config.d_model
        self.reward_head = RewardHead(d_model)

    def _get_rewards(self, model: nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
        """Extract scalar rewards from a model using the reward head."""
        x = model.embed(input_ids)
        freqs_cis = model.freqs_cis[: input_ids.shape[1]]
        for layer in model.layers:
            x, _, _ = layer(x, freqs_cis, mask=None, past_kv=None)
        x = model.norm(x)  # (B, T, d_model)
        return self.reward_head(x)  # (B,)

    def train_step(self, input_ids: torch.Tensor) -> dict:
        """Single training step: distillation + preference learning.

        The teacher is run with no_grad. The student and reward head are updated
        to match teacher soft distributions and prefer chosen over rejected sequences.

        Args:
            input_ids: (B, seq_len) input token ids.

        Returns:
            Dict with keys loss, distill_loss, pref_loss (all floats).
        """
        self.student_model.train()
        self.reward_head.train()
        self.optimizer.zero_grad()

        # Teacher rewards - no gradient
        with torch.no_grad():
            teacher_rewards = self._get_rewards(self.teacher_model, input_ids)

        # Student rewards - with gradient
        student_rewards = self._get_rewards(self.student_model, input_ids)

        # Distillation loss
        d_loss = distillation_loss(student_rewards, teacher_rewards, self.config.temperature)

        # Preference loss: split batch into chosen/rejected halves
        B = input_ids.shape[0]
        half = max(B // 2, 1)
        chosen_rewards = student_rewards[:half]
        rejected_rewards = student_rewards[half : half * 2] if B >= 2 else student_rewards[:half]
        p_loss = preference_loss(chosen_rewards, rejected_rewards, self.config.margin)

        # Combined loss
        loss = self.config.alpha * d_loss + (1.0 - self.config.alpha) * p_loss
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "distill_loss": d_loss.item(),
            "pref_loss": p_loss.item(),
        }
