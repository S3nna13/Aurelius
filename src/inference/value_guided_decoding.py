"""Value-Guided Decoding with token-level value functions.

A learned value function V(s_t) guides generation by scoring partial
sequences, enabling controlled decoding toward high-reward completions.

Reference: Mudgal et al. 2024 — https://arxiv.org/abs/2310.01542
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class TokenValueFunction(nn.Module):
    """Small MLP that maps hidden states to scalar value estimates.

    V(s_t) estimates the expected future reward from the current state.
    """

    def __init__(self, d_model: int, hidden_size: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, hidden_states: Tensor) -> Tensor:
        """Compute value estimates for all positions.

        Args:
            hidden_states: (B, T, d_model)

        Returns:
            (B, T) value estimates
        """
        return self.net(hidden_states).squeeze(-1)

    def value_at_last(self, hidden_states: Tensor) -> Tensor:
        """Return value estimate at the last token position.

        Args:
            hidden_states: (B, T, d_model)

        Returns:
            (B,) value at last position
        """
        return self.forward(hidden_states)[:, -1]


class ValueGuidedBeam:
    """Represents a single beam in value-guided search."""

    def __init__(
        self,
        token_ids: list[int],
        score: float,
        value_score: float,
        lm_score: float,
    ) -> None:
        self.token_ids = list(token_ids)
        self.score = score
        self.value_score = value_score
        self.lm_score = lm_score

    def extend(
        self,
        next_token: int,
        next_lm_logprob: float,
        next_value: float,
        alpha: float,
    ) -> ValueGuidedBeam:
        """Create new beam with token appended and updated scores.

        Args:
            next_token: token id to append
            next_lm_logprob: log probability of next_token from LM
            next_value: value estimate at the new position
            alpha: interpolation weight (1.0 = pure value, 0.0 = pure LM)

        Returns:
            New ValueGuidedBeam
        """
        new_lm_score = self.lm_score + next_lm_logprob
        new_value = next_value
        new_combined = alpha * new_value + (1.0 - alpha) * new_lm_score
        return ValueGuidedBeam(
            token_ids=self.token_ids + [next_token],
            score=new_combined,
            value_score=new_value,
            lm_score=new_lm_score,
        )


class ValueGuidedDecoder:
    """Decoder that uses a token-level value function to guide beam search."""

    def __init__(
        self,
        base_model_fn: Callable,
        value_fn: TokenValueFunction,
        beam_width: int = 4,
        alpha: float = 0.5,
    ) -> None:
        """
        Args:
            base_model_fn: callable (input_ids: LongTensor(1, T)) ->
                           (hidden: (1, T, d_model), logits: (1, T, vocab_size))
            value_fn: TokenValueFunction to score hidden states
            beam_width: number of beams to maintain
            alpha: interpolation weight between value (1.0) and LM (0.0)
        """
        self.base_model_fn = base_model_fn
        self.value_fn = value_fn
        self.beam_width = beam_width
        self.alpha = alpha

    @torch.no_grad()
    def generate(self, prompt_ids: Tensor, max_new_tokens: int) -> Tensor:
        """Generate tokens guided by value function.

        Args:
            prompt_ids: (1, T_prompt) input token ids
            max_new_tokens: number of new tokens to generate

        Returns:
            (max_new_tokens,) best generated token sequence
        """
        prompt_list = prompt_ids[0].tolist()
        beams: list[ValueGuidedBeam] = [ValueGuidedBeam(prompt_list, 0.0, 0.0, 0.0)]

        for _ in range(max_new_tokens):
            all_candidates: list[ValueGuidedBeam] = []

            for beam in beams:
                input_ids = torch.tensor(
                    [beam.token_ids], dtype=torch.long, device=prompt_ids.device
                )
                hidden, logits = self.base_model_fn(input_ids)

                # Log-probs at last position: (vocab_size,)
                last_logits = logits[0, -1, :]
                log_probs = F.log_softmax(last_logits, dim=-1)

                # Top-k candidates
                top_log_probs, top_tokens = torch.topk(log_probs, self.beam_width)

                # Value at last position: scalar
                value = self.value_fn.value_at_last(hidden).item()

                for k in range(self.beam_width):
                    tok = top_tokens[k].item()
                    lp = top_log_probs[k].item()
                    candidate = beam.extend(tok, lp, value, self.alpha)
                    all_candidates.append(candidate)

            # Prune to beam_width by combined score (descending)
            all_candidates.sort(key=lambda b: b.score, reverse=True)
            beams = all_candidates[: self.beam_width]

        best_beam = beams[0]
        new_tokens = best_beam.token_ids[len(prompt_list) :]
        # Ensure exactly max_new_tokens (trim or pad with 0)
        new_tokens = new_tokens[:max_new_tokens]
        while len(new_tokens) < max_new_tokens:
            new_tokens.append(0)

        return torch.tensor(new_tokens, dtype=torch.long, device=prompt_ids.device)


class ValueFunctionTrainer:
    """Trains the value function given (sequence, reward) pairs."""

    def __init__(
        self,
        value_fn: TokenValueFunction,
        optimizer: torch.optim.Optimizer,
        gamma: float = 0.99,
    ) -> None:
        self.value_fn = value_fn
        self.optimizer = optimizer
        self.gamma = gamma

    def compute_returns(self, rewards: Tensor) -> Tensor:
        """Compute discounted returns G_t = sum_{k=t}^{T} gamma^(k-t) * r_k.

        Args:
            rewards: (T,) per-step rewards

        Returns:
            (T,) discounted returns
        """
        T = rewards.shape[0]
        returns = torch.zeros_like(rewards)
        running = 0.0
        for t in range(T - 1, -1, -1):
            running = rewards[t].item() + self.gamma * running
            returns[t] = running
        return returns

    def train_step(self, hidden_states: Tensor, rewards: Tensor) -> dict[str, float]:
        """Perform one training step on the value function.

        Args:
            hidden_states: (1, T, d_model) from base model
            rewards: (T,) per-step rewards

        Returns:
            dict with keys 'loss', 'mean_value', 'mean_return'
        """
        returns = self.compute_returns(rewards)  # (T,)

        self.optimizer.zero_grad()
        # values: (1, T) -> (T,)
        values = self.value_fn(hidden_states)[0]
        loss = F.mse_loss(values, returns)
        loss.backward()
        self.optimizer.step()

        return {
            "loss": loss.item(),
            "mean_value": values.detach().mean().item(),
            "mean_return": returns.mean().item(),
        }
