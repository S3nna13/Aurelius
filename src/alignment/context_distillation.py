"""Context distillation: transfer system-prompted behavior into model weights.

Given a teacher (model + system_prompt) and student (same model, no prompt),
minimize KL(teacher_logits || student_logits) to bake in the system prompt.

Reference: Askell et al. 2021 "A General Language Assistant as a Laboratory
for Alignment" — section on distilling system prompts into the model's weights.

Usage:
    trainer = ContextDistillationTrainer(model, system_prompt_ids, cfg)
    trainer.train_step(user_message_ids, response_ids)
"""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW


@dataclass
class ContextDistillationConfig:
    temperature: float = 2.0        # KD temperature
    kl_weight: float = 1.0          # weight for KL loss vs CE loss
    ce_weight: float = 0.0          # weight for CE (response tokens) loss
    lr: float = 1e-5
    freeze_except_last_n: int = 4   # only fine-tune last N layers (efficiency)


def compute_teacher_logits(
    model: nn.Module,
    system_prompt_ids: torch.Tensor,  # (S_sys,)
    message_ids: torch.Tensor,        # (S_msg,)
    response_ids: torch.Tensor,       # (S_resp,)
) -> torch.Tensor:
    """Run model WITH system prompt prefix. Returns logits at response positions.

    Full sequence: [system_prompt | message | response]
    Returns logits at the response token positions: (S_resp, V)
    """
    # Concatenate: [sys_prompt | message | response]
    full_ids = torch.cat([system_prompt_ids, message_ids, response_ids], dim=0)  # (S_total,)
    input_ids = full_ids.unsqueeze(0)  # (1, S_total)

    with torch.no_grad():
        _, logits, _ = model(input_ids)  # (1, S_total, V)

    logits = logits.squeeze(0)  # (S_total, V)

    # Response positions start at offset = S_sys + S_msg
    # The model produces logits[t] predicting token t+1, but here we want
    # the logits that correspond to the response token positions.
    # Specifically, position (S_sys + S_msg - 1) predicts response[0], etc.
    s_prefix = system_prompt_ids.shape[0] + message_ids.shape[0]
    s_resp = response_ids.shape[0]
    # logits at positions [s_prefix-1 : s_prefix-1+s_resp] predict response tokens
    teacher_logits = logits[s_prefix - 1 : s_prefix - 1 + s_resp, :]  # (S_resp, V)
    return teacher_logits


def compute_student_logits(
    model: nn.Module,
    message_ids: torch.Tensor,   # (S_msg,)
    response_ids: torch.Tensor,  # (S_resp,)
) -> torch.Tensor:
    """Run model WITHOUT system prompt. Returns logits at response positions.

    Sequence: [message | response]
    Returns logits at response token positions: (S_resp, V)
    """
    # Concatenate: [message | response]
    full_ids = torch.cat([message_ids, response_ids], dim=0)  # (S_total,)
    input_ids = full_ids.unsqueeze(0)  # (1, S_total)

    _, logits, _ = model(input_ids)  # (1, S_total, V)
    logits = logits.squeeze(0)  # (S_total, V)

    # Response positions start at offset = S_msg
    # logits at positions [s_msg-1 : s_msg-1+s_resp] predict response tokens
    s_msg = message_ids.shape[0]
    s_resp = response_ids.shape[0]
    student_logits = logits[s_msg - 1 : s_msg - 1 + s_resp, :]  # (S_resp, V)
    return student_logits


def context_distillation_loss(
    student_logits: torch.Tensor,   # (S_resp, V)
    teacher_logits: torch.Tensor,   # (S_resp, V)
    response_ids: torch.Tensor,     # (S_resp,)
    cfg: ContextDistillationConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute context distillation loss.

    KL loss: KLDiv(log_softmax(student/T), softmax(teacher/T)) * T^2
    CE loss: cross_entropy(student_logits, response_ids) (if ce_weight > 0)
    Total: kl_weight * kl_loss + ce_weight * ce_loss

    Returns (total_loss, {"kl": float, "ce": float, "total": float})
    """
    T = cfg.temperature

    # Softened distributions for KL divergence
    student_log_soft = F.log_softmax(student_logits / T, dim=-1)  # (S_resp, V)
    teacher_soft = F.softmax(teacher_logits.detach() / T, dim=-1)  # (S_resp, V)

    # KL(teacher || student) with T^2 compensation
    kl_loss = F.kl_div(
        student_log_soft,
        teacher_soft,
        reduction="batchmean",
    ) * (T ** 2)

    # CE loss on raw student logits vs ground-truth response tokens
    ce_loss = F.cross_entropy(student_logits, response_ids)

    total_loss = cfg.kl_weight * kl_loss + cfg.ce_weight * ce_loss

    return total_loss, {
        "kl": kl_loss.item(),
        "ce": ce_loss.item(),
        "total": total_loss.item(),
    }


class ContextDistillationTrainer:
    """Trains a model to match system-prompted behavior without system prompt.

    The same model serves as both teacher (with sys prompt) and student (without).
    """

    def __init__(
        self,
        model: nn.Module,
        system_prompt_ids: torch.Tensor,  # (S_sys,) tokenized system prompt
        cfg: ContextDistillationConfig | None = None,
    ) -> None:
        self.model = model
        self.system_prompt_ids = system_prompt_ids
        self.cfg = cfg or ContextDistillationConfig()

        # Only fine-tune last N layers + lm_head
        trainable = []
        n_layers = len(list(model.layers))
        for i, layer in enumerate(model.layers):
            if i >= n_layers - self.cfg.freeze_except_last_n:
                trainable.extend(layer.parameters())
        if hasattr(model, 'lm_head'):
            trainable.extend(model.lm_head.parameters())

        self.optimizer = AdamW(trainable, lr=self.cfg.lr)

    def train_step(
        self,
        message_ids: torch.Tensor,   # (S_msg,)
        response_ids: torch.Tensor,  # (S_resp,)
    ) -> dict[str, float]:
        """One training step. Returns {"kl": float, "ce": float, "total": float}."""
        self.model.train()
        self.optimizer.zero_grad()

        # Teacher forward (no grad): model + system prompt
        teacher_logits = compute_teacher_logits(
            self.model,
            self.system_prompt_ids,
            message_ids,
            response_ids,
        )  # (S_resp, V)

        # Student forward (with grad): model without system prompt
        student_logits = compute_student_logits(
            self.model,
            message_ids,
            response_ids,
        )  # (S_resp, V)

        total_loss, metrics = context_distillation_loss(
            student_logits,
            teacher_logits,
            response_ids,
            self.cfg,
        )

        total_loss.backward()
        self.optimizer.step()

        return metrics
