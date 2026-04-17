"""Online/Iterative DPO -- on-policy preference learning.

Implements three loss variants:
- DPO  (Rafailov et al. 2023) -- log-sigmoid on implicit reward gap
- IPO  (Azar et al. 2023)     -- quadratic regression toward preference gap
- SLiC (Zhao et al. 2023)     -- hinge ranking + LM regulariser

Also includes OnlinePairGenerator (sample-then-score) and
OnlineDPOTrainer (end-to-end online loop with EMA ref updates).

Pure native PyTorch only; no external dependencies beyond stdlib + torch.
"""

from __future__ import annotations

import copy
from typing import Callable, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# OnlinePairGenerator
# ---------------------------------------------------------------------------

class OnlinePairGenerator:
    """Generate on-policy preference pairs by sampling then scoring.

    Candidates are drawn with temperature sampling from *model*.
    The highest-reward response becomes chosen; the lowest becomes
    rejected.  When every candidate achieves the same reward,
    ``generate_pair`` returns ``None`` so the caller can skip the batch.

    Args:
        model:        policy nn.Module; forward(input_ids) -> logits (B, T, V)
        reward_fn:    callable (generated_ids: Tensor) -> float or 0-d Tensor
        n_candidates: number of responses sampled per prompt
        temperature:  sampling temperature (> 0)
    """

    def __init__(
        self,
        model: nn.Module,
        reward_fn: Callable[[Tensor], float],
        n_candidates: int = 4,
        temperature: float = 0.8,
    ) -> None:
        self.model = model
        self.reward_fn = reward_fn
        self.n_candidates = n_candidates
        self.temperature = temperature

    # ------------------------------------------------------------------
    # Internal sampling
    # ------------------------------------------------------------------

    def _sample_one(self, input_ids: Tensor, max_new_tokens: int) -> Tensor:
        """Sample a single continuation autoregressively.

        Args:
            input_ids:      (1, T) prompt token ids
            max_new_tokens: number of tokens to generate

        Returns:
            generated: (1, T + max_new_tokens)
        """
        self.model.eval()
        current = input_ids.clone()
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self.model(current)           # (1, T', V)
                next_logits = logits[:, -1, :]         # (1, V)
                scaled = next_logits / max(self.temperature, 1e-8)
                probs = F.softmax(scaled, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)  # (1, 1)
                current = torch.cat([current, next_token], dim=1)
        return current

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate_pair(
        self,
        input_ids: Tensor,
        max_new_tokens: int = 16,
    ) -> Optional[tuple[Tensor, Tensor, float, float]]:
        """Generate a (chosen, rejected) pair from a single prompt.

        Args:
            input_ids:      (1, T) or (T,) prompt token ids
            max_new_tokens: tokens to append per candidate

        Returns:
            (chosen_ids, rejected_ids, chosen_reward, rejected_reward)
            or None when all candidates share the same reward.
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)

        candidates: list[Tensor] = []
        rewards: list[float] = []

        for _ in range(self.n_candidates):
            gen = self._sample_one(input_ids, max_new_tokens)
            reward_val = self.reward_fn(gen)
            if isinstance(reward_val, Tensor):
                reward_val = reward_val.item()
            candidates.append(gen)
            rewards.append(float(reward_val))

        best_idx = int(max(range(len(rewards)), key=lambda i: rewards[i]))
        worst_idx = int(min(range(len(rewards)), key=lambda i: rewards[i]))

        if rewards[best_idx] == rewards[worst_idx]:
            return None

        return (
            candidates[best_idx],
            candidates[worst_idx],
            rewards[best_idx],
            rewards[worst_idx],
        )

    def batch_generate(
        self,
        prompts: list[Tensor],
    ) -> list[Optional[tuple[Tensor, Tensor, float, float]]]:
        """Generate pairs for a list of prompts.

        Args:
            prompts: list of (1, T) or (T,) tensors

        Returns:
            list of (chosen, rejected, chosen_reward, rejected_reward) or None
            per prompt (None when all candidates tied).
        """
        return [self.generate_pair(p) for p in prompts]


# ---------------------------------------------------------------------------
# IPOLoss
# ---------------------------------------------------------------------------

class IPOLoss(nn.Module):
    """Identity Preference Optimization loss (Azar et al. 2023).

    Unlike DPO, IPO regresses the implicit reward gap toward a target of
    1/(2*tau) using a quadratic objective.  This avoids reward over-
    optimisation caused by DPO's unbounded log-sigmoid.

    Loss = mean( (h_w - h_l - 1/(2*tau))^2 )

    where
        h_w = log( pi_theta(y_w) / pi_ref(y_w) )
        h_l = log( pi_theta(y_l) / pi_ref(y_l) )

    Args:
        tau: preference-gap regularisation strength (default 0.1)
    """

    def __init__(self, tau: float = 0.1) -> None:
        super().__init__()
        self.tau = tau

    def forward(
        self,
        policy_chosen_logps: Tensor,
        policy_rejected_logps: Tensor,
        ref_chosen_logps: Tensor,
        ref_rejected_logps: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute IPO loss and implicit rewards.

        Args:
            policy_chosen_logps:   (B,) log pi_theta(y_w | x)
            policy_rejected_logps: (B,) log pi_theta(y_l | x)
            ref_chosen_logps:      (B,) log pi_ref(y_w | x)
            ref_rejected_logps:    (B,) log pi_ref(y_l | x)

        Returns:
            loss:       scalar IPO loss
            h_chosen:   (B,) implicit reward for chosen (with grad)
            h_rejected: (B,) implicit reward for rejected (with grad)
        """
        h_chosen = policy_chosen_logps - ref_chosen_logps
        h_rejected = policy_rejected_logps - ref_rejected_logps

        target = 1.0 / (2.0 * self.tau)
        loss = ((h_chosen - h_rejected - target) ** 2).mean()

        return loss, h_chosen, h_rejected


# ---------------------------------------------------------------------------
# SLiCLoss
# ---------------------------------------------------------------------------

class SLiCLoss(nn.Module):
    """Sequence Likelihood Calibration loss (Zhao et al. 2023).

    Combines a hinge ranking loss (prefers chosen over rejected by *delta*)
    with a language-modelling regulariser to prevent policy drift.

    Args:
        delta:     hinge margin (default 1.0)
        lm_weight: weight on the LM regulariser (default 0.1)
    """

    def __init__(self, delta: float = 1.0, lm_weight: float = 0.1) -> None:
        super().__init__()
        self.delta = delta
        self.lm_weight = lm_weight

    def forward(
        self,
        policy_chosen_logps: Tensor,
        policy_rejected_logps: Tensor,
        input_ids: Tensor,
        policy_logits: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Compute SLiC loss.

        Args:
            policy_chosen_logps:   (B,) log pi_theta(y_w | x)
            policy_rejected_logps: (B,) log pi_theta(y_l | x)
            input_ids:             (B, T) token ids for LM regulariser
            policy_logits:         (B, T, V) logits from policy for input_ids

        Returns:
            total_loss: scalar  rank_loss + reg_loss
            rank_loss:  scalar  hinge ranking loss
            reg_loss:   scalar  LM regularisation loss
        """
        margin = policy_chosen_logps - policy_rejected_logps
        rank_loss = torch.clamp(self.delta - margin, min=0.0).mean()

        shift_logits = policy_logits[:, :-1, :].contiguous()
        shift_labels = input_ids[:, 1:].contiguous()
        reg_loss = self.lm_weight * F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
        )

        return rank_loss + reg_loss, rank_loss, reg_loss


# ---------------------------------------------------------------------------
# DPOVariantTrainer
# ---------------------------------------------------------------------------

class DPOVariantTrainer:
    """Unified trainer supporting DPO, IPO, and SLiC loss objectives.

    Args:
        policy_model: trainable policy nn.Module
        ref_model:    frozen reference nn.Module (will be frozen internally)
        optimizer:    torch Optimizer for policy_model
        loss_type:    one of "dpo", "ipo", "slic"
        beta:         KL / tau strength; used as DPO beta or IPO tau
    """

    _VALID_LOSS_TYPES = {"dpo", "ipo", "slic"}

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_type: str = "dpo",
        beta: float = 0.1,
    ) -> None:
        if loss_type not in self._VALID_LOSS_TYPES:
            raise ValueError(
                f"loss_type must be one of {self._VALID_LOSS_TYPES}, got '{loss_type}'"
            )
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.optimizer = optimizer
        self.loss_type = loss_type
        self.beta = beta

        # Freeze ref model
        for param in self.ref_model.parameters():
            param.requires_grad_(False)
        self.ref_model.eval()

        self._ipo_loss = IPOLoss(tau=beta)
        self._slic_loss = SLiCLoss(delta=1.0, lm_weight=0.1)

    # ------------------------------------------------------------------
    # Log-prob computation
    # ------------------------------------------------------------------

    def compute_logps(
        self, model: nn.Module, input_ids: Tensor, labels: Tensor
    ) -> Tensor:
        """Compute per-sequence summed log-probs.

        Args:
            model:     nn.Module; forward(input_ids) -> logits (B, T, V)
            input_ids: (B, T)
            labels:    (B, T)

        Returns:
            log_probs: (B,)
        """
        logits = model(input_ids)
        shift_logits = logits[:, :-1, :].contiguous()
        shift_labels = labels[:, 1:].contiguous()

        log_probs_all = F.log_softmax(shift_logits, dim=-1)
        gather_labels = shift_labels.clone()
        gather_labels[gather_labels == -100] = 0
        token_logps = log_probs_all.gather(
            dim=-1, index=gather_labels.unsqueeze(-1)
        ).squeeze(-1)

        mask = (shift_labels != -100).float()
        return (token_logps * mask).sum(dim=-1)

    # ------------------------------------------------------------------
    # DPO helper
    # ------------------------------------------------------------------

    def _dpo_loss(
        self,
        policy_chosen_logps: Tensor,
        policy_rejected_logps: Tensor,
        ref_chosen_logps: Tensor,
        ref_rejected_logps: Tensor,
    ) -> Tensor:
        chosen_reward = self.beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_reward = self.beta * (policy_rejected_logps - ref_rejected_logps)
        return -F.logsigmoid(chosen_reward - rejected_reward).mean()

    # ------------------------------------------------------------------
    # train_step
    # ------------------------------------------------------------------

    def train_step(
        self,
        chosen_ids: Tensor,
        chosen_labels: Tensor,
        rejected_ids: Tensor,
        rejected_labels: Tensor,
    ) -> dict:
        """Run one optimisation step.

        Args:
            chosen_ids:      (B, T) preferred sequence token ids
            chosen_labels:   (B, T) labels for preferred seq
            rejected_ids:    (B, T) dis-preferred sequence token ids
            rejected_labels: (B, T) labels for dis-preferred seq

        Returns:
            dict with keys: loss, implicit_reward_diff, accuracy
        """
        self.policy_model.train()
        self.optimizer.zero_grad()

        policy_chosen_logps = self.compute_logps(
            self.policy_model, chosen_ids, chosen_labels
        )
        policy_rejected_logps = self.compute_logps(
            self.policy_model, rejected_ids, rejected_labels
        )

        with torch.no_grad():
            ref_chosen_logps = self.compute_logps(
                self.ref_model, chosen_ids, chosen_labels
            )
            ref_rejected_logps = self.compute_logps(
                self.ref_model, rejected_ids, rejected_labels
            )

        if self.loss_type == "dpo":
            loss = self._dpo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
            )
            h_chosen = self.beta * (policy_chosen_logps - ref_chosen_logps)
            h_rejected = self.beta * (policy_rejected_logps - ref_rejected_logps)

        elif self.loss_type == "ipo":
            loss, h_chosen, h_rejected = self._ipo_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                ref_chosen_logps,
                ref_rejected_logps,
            )

        else:  # slic
            policy_chosen_logits = self.policy_model(chosen_ids)
            loss, _, _ = self._slic_loss(
                policy_chosen_logps,
                policy_rejected_logps,
                chosen_ids,
                policy_chosen_logits,
            )
            h_chosen = policy_chosen_logps - ref_chosen_logps
            h_rejected = policy_rejected_logps - ref_rejected_logps

        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_model.parameters(), max_norm=1.0)
        self.optimizer.step()

        with torch.no_grad():
            implicit_reward_diff = (h_chosen - h_rejected).mean()
            accuracy = (h_chosen > h_rejected).float().mean()

        return {
            "loss": loss.detach(),
            "implicit_reward_diff": implicit_reward_diff.detach(),
            "accuracy": accuracy.detach(),
        }


# ---------------------------------------------------------------------------
# OnlineDPOTrainer
# ---------------------------------------------------------------------------

class OnlineDPOTrainer:
    """Full online DPO training loop.

    Each ``online_step`` generates preference pairs on-the-fly from the
    current policy, filters out ties, then calls ``DPOVariantTrainer``.
    ``ref_update_step`` performs an EMA update of the reference model.

    Args:
        policy_model:    trainable policy nn.Module
        ref_model:       reference nn.Module (frozen internally)
        reward_fn:       callable (generated_ids: Tensor) -> float / 0-d Tensor
        optimizer:       torch Optimizer for policy_model
        pair_generator:  OnlinePairGenerator instance
        loss_type:       "dpo", "ipo", or "slic"
    """

    def __init__(
        self,
        policy_model: nn.Module,
        ref_model: nn.Module,
        reward_fn: Callable[[Tensor], float],
        optimizer: torch.optim.Optimizer,
        pair_generator: OnlinePairGenerator,
        loss_type: str = "dpo",
    ) -> None:
        self.policy_model = policy_model
        self.ref_model = ref_model
        self.reward_fn = reward_fn
        self.optimizer = optimizer
        self.pair_generator = pair_generator
        self.loss_type = loss_type

        self.variant_trainer = DPOVariantTrainer(
            policy_model=policy_model,
            ref_model=ref_model,
            optimizer=optimizer,
            loss_type=loss_type,
        )

    # ------------------------------------------------------------------
    # online_step
    # ------------------------------------------------------------------

    def online_step(
        self,
        prompts: Tensor,
        max_new_tokens: int = 8,
    ) -> dict:
        """One online step: generate pairs, filter ties, train.

        Args:
            prompts:        (B, T) prompt token ids -- each row is one prompt
            max_new_tokens: tokens to generate per candidate

        Returns:
            dict with keys: loss, n_valid_pairs, mean_reward_gap
        """
        batch_size = prompts.size(0)
        prompt_list = [prompts[i].unsqueeze(0) for i in range(batch_size)]

        raw_pairs = [
            self.pair_generator.generate_pair(p, max_new_tokens=max_new_tokens)
            for p in prompt_list
        ]

        valid_pairs = [p for p in raw_pairs if p is not None]
        n_valid = len(valid_pairs)

        if n_valid == 0:
            return {
                "loss": torch.tensor(0.0),
                "n_valid_pairs": 0,
                "mean_reward_gap": torch.tensor(0.0),
            }

        chosen_list: list[Tensor] = []
        rejected_list: list[Tensor] = []
        reward_gaps: list[float] = []

        for chosen_ids, rejected_ids, chosen_r, rejected_r in valid_pairs:
            chosen_list.append(chosen_ids.squeeze(0))
            rejected_list.append(rejected_ids.squeeze(0))
            reward_gaps.append(chosen_r - rejected_r)

        max_len_c = max(t.size(0) for t in chosen_list)
        max_len_r = max(t.size(0) for t in rejected_list)
        max_len = max(max_len_c, max_len_r)

        def _pad(seq: Tensor, target_len: int) -> Tensor:
            if seq.size(0) < target_len:
                pad = torch.zeros(
                    target_len - seq.size(0), dtype=seq.dtype, device=seq.device
                )
                seq = torch.cat([seq, pad], dim=0)
            return seq

        chosen_batch = torch.stack(
            [_pad(t, max_len) for t in chosen_list], dim=0
        )
        rejected_batch = torch.stack(
            [_pad(t, max_len) for t in rejected_list], dim=0
        )

        step_result = self.variant_trainer.train_step(
            chosen_ids=chosen_batch,
            chosen_labels=chosen_batch.clone(),
            rejected_ids=rejected_batch,
            rejected_labels=rejected_batch.clone(),
        )

        mean_reward_gap = torch.tensor(
            sum(reward_gaps) / len(reward_gaps), dtype=torch.float32
        )

        return {
            "loss": step_result["loss"],
            "n_valid_pairs": n_valid,
            "mean_reward_gap": mean_reward_gap,
        }

    # ------------------------------------------------------------------
    # ref_update_step
    # ------------------------------------------------------------------

    def ref_update_step(self, alpha: float = 0.1) -> None:
        """EMA update of ref_model toward policy_model.

        For each parameter:
            ref_param = alpha * policy_param + (1 - alpha) * ref_param

        Args:
            alpha: interpolation weight; 0 means no change, 1 means full copy
        """
        with torch.no_grad():
            for ref_param, policy_param in zip(
                self.ref_model.parameters(), self.policy_model.parameters()
            ):
                ref_param.data.mul_(1.0 - alpha).add_(
                    policy_param.data, alpha=alpha
                )
