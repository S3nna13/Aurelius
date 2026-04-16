"""Stepwise Direct Preference Optimization (Stepwise DPO).

Implements step-level preference optimization for multi-step reasoning tasks.
Rather than assigning a single preference signal to the full response, each
reasoning step is individually labeled as correct/incorrect, enabling
fine-grained credit assignment (related to Math-Shepherd and Step-DPO).

References:
    Math-Shepherd: Verify-Then-Compare (Lightman et al. 2023)
    Step-DPO: Step-level Preference Optimization (Lai et al. 2024)
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ReasoningStep:
    """A single reasoning step with its token ids and correctness label.

    Args:
        step_ids: Token ids for this step (1-D tensor of length step_len).
        is_correct: Whether this step is labeled as correct.
        step_reward: Optional scalar reward for this step. Defaults to 0.0.
    """

    step_ids: Tensor
    is_correct: bool
    step_reward: float = 0.0


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class StepwiseDPOConfig:
    """Configuration for Stepwise DPO training.

    Args:
        beta: KL penalty coefficient (same as standard DPO).
        step_weight_decay: Exponential decay factor for step weights.
            Later steps (closer to the final answer) receive lower weight.
            Weight for step i: decay^(n_steps - 1 - i), so step 0 (earliest)
            has the highest weight when decay < 1.
        normalize_weights: Whether to normalize step weights to sum to 1.
        min_step_weight: Minimum per-step weight after normalization.
        aggregate_method: How to aggregate per-step losses.
            'weighted_sum': sum of (weight * step_loss).
    """

    beta: float = 0.1
    step_weight_decay: float = 0.9
    normalize_weights: bool = True
    min_step_weight: float = 0.01
    aggregate_method: str = "weighted_sum"


# ---------------------------------------------------------------------------
# Standalone helper functions
# ---------------------------------------------------------------------------

def parse_reasoning_steps(
    response_ids: Tensor,
    separator_id: int,
) -> List[Tensor]:
    """Split a response tensor into individual reasoning steps.

    Splits at positions where the token equals separator_id. The separator
    tokens themselves are excluded from the returned step tensors.
    If no separator is found, the entire response is returned as a single step.

    Args:
        response_ids: 1-D tensor of token ids representing the full response.
        separator_id: Token id used as a step boundary marker.

    Returns:
        List of 1-D tensors, one per step, excluding separator tokens.
    """
    if response_ids.dim() != 1:
        raise ValueError(
            f"response_ids must be 1-D, got shape {tuple(response_ids.shape)}"
        )

    sep_positions = (response_ids == separator_id).nonzero(as_tuple=False).squeeze(-1)

    if sep_positions.numel() == 0:
        # No separator found - return full response as a single step
        return [response_ids]

    steps: List[Tensor] = []
    prev = 0
    for pos in sep_positions.tolist():
        steps.append(response_ids[prev:pos])
        prev = pos + 1  # skip separator token

    # Append the final segment after the last separator
    steps.append(response_ids[prev:])
    return steps


def label_steps_by_prefix_match(
    steps: List[Tensor],
    correct_steps: List[Tensor],
) -> List[bool]:
    """Label each step as correct if its first token matches the corresponding
    correct step's first token.

    For positions beyond the length of correct_steps, labels default to False.

    Args:
        steps: List of 1-D step tensors to label.
        correct_steps: List of 1-D reference (correct) step tensors.

    Returns:
        List of bool labels, one per element in steps.
    """
    labels: List[bool] = []
    for i, step in enumerate(steps):
        if i >= len(correct_steps):
            labels.append(False)
            continue
        ref = correct_steps[i]
        if step.numel() == 0 or ref.numel() == 0:
            labels.append(False)
        else:
            labels.append(bool(step[0].item() == ref[0].item()))
    return labels


# ---------------------------------------------------------------------------
# Stepwise DPO Trainer
# ---------------------------------------------------------------------------

class StepwiseDPOTrainer:
    """Trainer for Stepwise Direct Preference Optimization.

    Assigns DPO-style preference learning at the step level. Each reasoning
    step receives an independent DPO loss, weighted by its position-based
    importance. Earlier steps that are critical for reaching the correct
    answer can be weighted more or less strongly via step_weight_decay.

    Args:
        policy: Policy model. Forward call must accept (B, T) input_ids and
                return (B, T, V) logits or a compatible tuple/list.
        ref_policy: Frozen reference model with the same interface.
        beta: KL penalty coefficient.
        step_weight_decay: Exponential decay applied to step weights.
            Step weight at index i = decay^(n_steps - 1 - i), so earlier
            steps (lower i) receive higher weight when decay < 1.
    """

    def __init__(
        self,
        policy: nn.Module,
        ref_policy: nn.Module,
        beta: float = 0.1,
        step_weight_decay: float = 0.9,
    ) -> None:
        self.policy = policy
        self.ref_policy = ref_policy
        self.beta = beta
        self.step_weight_decay = step_weight_decay

        # Freeze and eval the reference model
        for p in self.ref_policy.parameters():
            p.requires_grad_(False)
        self.ref_policy.eval()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_logits(self, model: nn.Module, input_ids: Tensor) -> Tensor:
        """Extract (B, T, V) or (T, V) logits from a model forward pass."""
        output = model(input_ids)
        if isinstance(output, Tensor):
            return output
        if isinstance(output, (tuple, list)):
            for idx in (1, 0):
                if idx < len(output) and isinstance(output[idx], Tensor):
                    candidate = output[idx]
                    if candidate.dim() in (2, 3):
                        return candidate
            for item in output:
                if isinstance(item, Tensor) and item.dim() in (2, 3):
                    return item
        raise ValueError(
            f"Cannot extract logits from model output of type {type(output)}. "
            "Expected a (B, T, V) or (T, V) tensor or a tuple/list containing one."
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_step_weights(self, n_steps: int) -> Tensor:
        """Return a (n_steps,) weight tensor using exponential decay.

        Weight for step i = decay^(n_steps - i), following the spec formula
        Step weight: decay^(n_steps - step_idx). With decay < 1:
          - step 0 (earliest): exponent = n_steps (smallest weight)
          - step n_steps-1 (latest, closest to answer): exponent = 1 (largest weight)

        This means later steps receive more credit (higher weight).
        The returned weights are normalized to sum to 1.

        Args:
            n_steps: Number of reasoning steps.

        Returns:
            (n_steps,) float tensor, normalized to sum to 1.0.
        """
        # exponents[i] = n_steps - i  =>  arange from n_steps down to 1
        exponents = torch.arange(n_steps, 0, -1, dtype=torch.float32)
        weights = torch.pow(
            torch.tensor(self.step_weight_decay, dtype=torch.float32), exponents
        )
        # Normalize so weights sum to 1
        weights = weights / weights.sum().clamp(min=1e-8)
        return weights

    def compute_step_log_probs(
        self,
        model: nn.Module,
        step_ids_list: List[Tensor],
        context_ids: Tensor,
    ) -> List[Tensor]:
        """Compute log probabilities for each step conditioned on context + prior steps.

        For step k, the input to the model is:
            [context_ids, step_0, step_1, ..., step_{k-1}, step_k]
        and we compute log probs for the tokens of step_k (the final segment).

        Args:
            model: The language model.
            step_ids_list: List of 1-D token id tensors, one per step.
            context_ids: 1-D tensor of context (prompt) token ids.

        Returns:
            List of (step_len,) log probability tensors, one per step.
        """
        step_logps: List[Tensor] = []

        # Build cumulative prefix: context + steps[0..k-1]
        prefix = context_ids  # (prefix_len,)

        for step_ids in step_ids_list:
            step_len = step_ids.size(0)

            if step_len == 0:
                # Zero-length step: return empty tensor
                step_logps.append(torch.tensor([], dtype=torch.float32))
                continue

            # Full input: prefix + current step
            full_input = torch.cat([prefix, step_ids], dim=0)  # (prefix_len + step_len,)
            # Add batch dimension
            input_2d = full_input.unsqueeze(0)  # (1, T)

            logits = self._get_logits(model, input_2d)  # (1, T, V) or (T, V)
            if logits.dim() == 3:
                logits = logits.squeeze(0)  # (T, V)

            log_probs_all = F.log_softmax(logits, dim=-1)  # (T, V)

            # We want log P(step_ids[t] | context + step_ids[:t])
            # That corresponds to positions [prefix_len-1 .. prefix_len+step_len-2] in logits
            # (shifted left by 1 for next-token prediction)
            prefix_len = prefix.size(0)
            # logits at position j predicts token j+1
            # step token at index t in step_ids is at position prefix_len + t in full_input
            # => predicted by logit at position prefix_len + t - 1
            start = prefix_len - 1
            end = prefix_len + step_len - 1  # exclusive

            relevant_logps = log_probs_all[start:end]  # (step_len, V)
            target_ids = step_ids  # (step_len,)

            token_logps = relevant_logps.gather(
                -1, target_ids.unsqueeze(-1)
            ).squeeze(-1)  # (step_len,)

            step_logps.append(token_logps)

            # Extend prefix to include current step for the next iteration
            prefix = full_input

        return step_logps

    def compute_stepwise_dpo_loss(
        self,
        chosen_steps: List[ReasoningStep],
        rejected_steps: List[ReasoningStep],
        chosen_step_logps: List[Tensor],
        rejected_step_logps: List[Tensor],
        ref_chosen_logps: List[Tensor],
        ref_rejected_logps: List[Tensor],
    ) -> Tuple[Tensor, Dict]:
        """Compute the stepwise DPO loss across all step pairs.

        For each step index i, computes a standard DPO loss using the
        sequence-summed log probs for that step, then weights it by the
        step importance weight.

        Args:
            chosen_steps: List of ReasoningStep for the chosen (preferred) response.
            rejected_steps: List of ReasoningStep for the rejected response.
            chosen_step_logps: Per-step log prob tensors for chosen under policy.
            rejected_step_logps: Per-step log prob tensors for rejected under policy.
            ref_chosen_logps: Per-step log prob tensors for chosen under reference.
            ref_rejected_logps: Per-step log prob tensors for rejected under reference.

        Returns:
            Tuple of (total_loss, metrics_dict). metrics_dict keys:
                - loss: float
                - mean_chosen_reward: float
                - mean_rejected_reward: float
                - reward_accuracy: float
                - n_steps: int
                - step_weights: list of float
        """
        n_steps = min(
            len(chosen_steps),
            len(rejected_steps),
            len(chosen_step_logps),
            len(rejected_step_logps),
            len(ref_chosen_logps),
            len(ref_rejected_logps),
        )

        if n_steps == 0:
            zero = torch.tensor(0.0, requires_grad=True)
            return zero, {
                "loss": 0.0,
                "mean_chosen_reward": 0.0,
                "mean_rejected_reward": 0.0,
                "reward_accuracy": 0.0,
                "n_steps": 0,
                "step_weights": [],
            }

        weights = self.compute_step_weights(n_steps)  # (n_steps,)

        step_losses: List[Tensor] = []
        chosen_rewards_list: List[float] = []
        rejected_rewards_list: List[float] = []

        for i in range(n_steps):
            # Aggregate log probs over the step (sum over tokens)
            c_logp = chosen_step_logps[i]
            r_logp = rejected_step_logps[i]
            rc_logp = ref_chosen_logps[i]
            rr_logp = ref_rejected_logps[i]

            pi_chosen = c_logp.sum() if c_logp.numel() > 0 else torch.tensor(0.0)
            pi_rejected = r_logp.sum() if r_logp.numel() > 0 else torch.tensor(0.0)
            ref_chosen = rc_logp.sum() if rc_logp.numel() > 0 else torch.tensor(0.0)
            ref_rejected = rr_logp.sum() if rr_logp.numel() > 0 else torch.tensor(0.0)

            chosen_reward = self.beta * (pi_chosen - ref_chosen)
            rejected_reward = self.beta * (pi_rejected - ref_rejected)
            reward_diff = chosen_reward - rejected_reward

            step_loss = -F.logsigmoid(reward_diff)

            step_losses.append(step_loss)
            chosen_rewards_list.append(chosen_reward.detach().item())
            rejected_rewards_list.append(rejected_reward.detach().item())

        # Stack losses and apply weights: (n_steps,)
        losses_tensor = torch.stack(step_losses)  # (n_steps,)
        weights_device = weights.to(losses_tensor.device)
        total_loss = (losses_tensor * weights_device).sum()

        chosen_r = torch.tensor(chosen_rewards_list)
        rejected_r = torch.tensor(rejected_rewards_list)
        accuracy = float((chosen_r > rejected_r).float().mean().item())

        metrics: Dict = {
            "loss": total_loss.detach().item(),
            "mean_chosen_reward": float(chosen_r.mean().item()),
            "mean_rejected_reward": float(rejected_r.mean().item()),
            "reward_accuracy": accuracy,
            "n_steps": n_steps,
            "step_weights": weights.tolist(),
        }

        return total_loss, metrics

    def train_step(
        self,
        chosen_steps: List[ReasoningStep],
        rejected_steps: List[ReasoningStep],
        context_ids: Tensor,
    ) -> Dict:
        """Execute a single training forward pass and return metrics.

        Computes log probs under both policy and reference models, then
        computes the stepwise DPO loss. Does NOT call backward() or update
        parameters - the caller is responsible for optimizer steps.

        Args:
            chosen_steps: Reasoning steps for the chosen (preferred) response.
            rejected_steps: Reasoning steps for the rejected response.
            context_ids: 1-D context (prompt) token ids shared by both responses.

        Returns:
            Metrics dict with at minimum a 'loss' key (scalar tensor).
        """
        chosen_ids_list = [s.step_ids for s in chosen_steps]
        rejected_ids_list = [s.step_ids for s in rejected_steps]

        # Policy log probs (with grad)
        self.policy.train()
        chosen_step_logps = self.compute_step_log_probs(
            self.policy, chosen_ids_list, context_ids
        )
        rejected_step_logps = self.compute_step_log_probs(
            self.policy, rejected_ids_list, context_ids
        )

        # Reference log probs (no grad)
        with torch.no_grad():
            ref_chosen_logps = self.compute_step_log_probs(
                self.ref_policy, chosen_ids_list, context_ids
            )
            ref_rejected_logps = self.compute_step_log_probs(
                self.ref_policy, rejected_ids_list, context_ids
            )

        loss, metrics = self.compute_stepwise_dpo_loss(
            chosen_steps=chosen_steps,
            rejected_steps=rejected_steps,
            chosen_step_logps=chosen_step_logps,
            rejected_step_logps=rejected_step_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
        )

        metrics["loss"] = loss  # return tensor so caller can call .backward()
        return metrics
