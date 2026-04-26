"""Chain-of-Thought Distillation (Ho et al. 2022 "Large Language Models are Reasoning Teachers").

Distills reasoning chains from a large teacher model into a small student model by
training the student to produce chain-of-thought rationales that mimic the teacher's.

Combined loss: alpha * L_cot + (1-alpha) * L_answer
With optional KD soft targets from teacher logits.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class CoTDistillConfig:
    alpha: float = 0.5  # weight for CoT loss (1-alpha for answer loss)
    temperature: float = 2.0  # KD temperature for soft targets
    lr: float = 1e-4
    max_seq_len: int = 64
    n_steps: int = 4
    cot_loss_type: str = "ce"  # "ce" (cross-entropy) or "kl" (KL from teacher logits)
    answer_loss_weight: float = 1.0


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass
class CoTSample:
    """A training sample with reasoning chain."""

    question_ids: Tensor  # (T_q,) tokenized question
    reasoning_ids: Tensor  # (T_r,) tokenized chain-of-thought
    answer_ids: Tensor  # (T_a,) tokenized final answer
    teacher_logits: Tensor | None = None  # (T_r + T_a, vocab) optional teacher soft targets


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------


def cot_cross_entropy_loss(
    student_logits: Tensor,  # (B, T, vocab)
    target_ids: Tensor,  # (B, T)
    reasoning_mask: Tensor,  # (B, T) 1 for CoT tokens, 0 for answer tokens
    alpha: float = 0.5,
) -> tuple[Tensor, dict[str, float]]:
    """Weighted CE: alpha * L_cot + (1-alpha) * L_answer.

    Returns (loss, {'cot_loss': float, 'answer_loss': float, 'total_loss': float})
    """
    B, T, V = student_logits.shape

    # Flatten for per-token CE
    logits_flat = student_logits.reshape(-1, V)  # (B*T, vocab)
    targets_flat = target_ids.reshape(-1)  # (B*T,)
    reasoning_mask_flat = reasoning_mask.reshape(-1).float()  # (B*T,)
    answer_mask_flat = 1.0 - reasoning_mask_flat  # (B*T,)

    # Per-token CE (no reduction)
    per_token_ce = F.cross_entropy(logits_flat, targets_flat, reduction="none")  # (B*T,)

    # CoT loss: average over reasoning positions
    cot_denom = reasoning_mask_flat.sum().clamp(min=1.0)
    cot_loss = (per_token_ce * reasoning_mask_flat).sum() / cot_denom

    # Answer loss: average over answer positions
    answer_denom = answer_mask_flat.sum().clamp(min=1.0)
    answer_loss = (per_token_ce * answer_mask_flat).sum() / answer_denom

    total_loss = alpha * cot_loss + (1.0 - alpha) * answer_loss

    metrics = {
        "cot_loss": cot_loss.item(),
        "answer_loss": answer_loss.item(),
        "total_loss": total_loss.item(),
    }
    return total_loss, metrics


def knowledge_distillation_loss(
    student_logits: Tensor,  # (B, T, vocab)
    teacher_logits: Tensor,  # (B, T, vocab)
    target_ids: Tensor,  # (B, T)
    temperature: float = 2.0,
    alpha: float = 0.5,
) -> Tensor:
    """KD loss: alpha * KL(student || teacher) + (1-alpha) * CE(student, targets).

    Both logits are divided by temperature for soft label distillation.
    Returns scalar loss.
    """
    B, T, V = student_logits.shape

    # Hard CE loss
    hard_loss = F.cross_entropy(
        student_logits.reshape(-1, V),
        target_ids.reshape(-1),
    )

    # Soft KL loss
    student_log_soft = F.log_softmax(student_logits / temperature, dim=-1)  # (B, T, V)
    teacher_soft = F.softmax(teacher_logits.detach() / temperature, dim=-1)  # (B, T, V)

    kl_loss = F.kl_div(
        student_log_soft.reshape(-1, V),
        teacher_soft.reshape(-1, V),
        reduction="batchmean",
    ) * (temperature**2)  # T^2 compensation for reduced gradient magnitude

    total_loss = alpha * kl_loss + (1.0 - alpha) * hard_loss
    return total_loss


# ---------------------------------------------------------------------------
# Batch preparation
# ---------------------------------------------------------------------------


def prepare_cot_batch(
    samples: list[CoTSample],
    max_len: int = 64,
    pad_token: int = 0,
) -> dict[str, Tensor]:
    """Pad and collate CoT samples into batch tensors.

    Concatenates [question | reasoning | answer] tokens per sample, then pads
    to the same length. Target ids are shifted by 1 for next-token prediction.

    Returns:
        {
          'input_ids':      (B, T),
          'target_ids':     (B, T),  # shifted by 1 for next-token prediction
          'reasoning_mask': (B, T),  # 1 for CoT positions
          'attention_mask': (B, T),  # 0 for padding
        }
    """
    input_ids_list: list[Tensor] = []
    target_ids_list: list[Tensor] = []
    reasoning_mask_list: list[Tensor] = []
    attention_mask_list: list[Tensor] = []

    for sample in samples:
        q = sample.question_ids
        r = sample.reasoning_ids
        a = sample.answer_ids

        # Full sequence = question + reasoning + answer
        full_seq = torch.cat([q, r, a], dim=0)  # (T_q + T_r + T_a,)

        # Truncate to max_len + 1 so we can create shifted targets
        max_raw = max_len + 1
        if full_seq.shape[0] > max_raw:
            full_seq = full_seq[:max_raw]

        # input = all tokens except last; target = all tokens except first (shifted)
        inp = full_seq[:-1]  # (T-1,)
        tgt = full_seq[1:]  # (T-1,)
        seq_len = inp.shape[0]

        # Build reasoning mask for the *input* positions.
        # Question spans [0, T_q) in full_seq; reasoning spans [T_q, T_q+T_r).
        # After the shift, reasoning positions in inp are [T_q-1, T_q+T_r-1)
        # but we keep it simple: mark positions where the *input* token is a
        # reasoning token, i.e. [T_q, T_q+T_r) clipped to seq_len.
        T_q = min(q.shape[0], seq_len)
        T_r = r.shape[0]
        reasoning_start = T_q
        reasoning_end = min(T_q + T_r, seq_len)

        rmask = torch.zeros(seq_len, dtype=torch.long)
        if reasoning_start < reasoning_end:
            rmask[reasoning_start:reasoning_end] = 1

        # Pad to max_len
        pad_len = max_len - seq_len
        if pad_len > 0:
            pad_tensor = torch.full((pad_len,), pad_token, dtype=inp.dtype)
            zero_pad = torch.zeros(pad_len, dtype=torch.long)
            inp = torch.cat([inp, pad_tensor])
            tgt = torch.cat([tgt, pad_tensor])
            rmask = torch.cat([rmask, zero_pad])

        attn_mask = torch.zeros(max_len, dtype=torch.long)
        attn_mask[:seq_len] = 1

        input_ids_list.append(inp[:max_len])
        target_ids_list.append(tgt[:max_len])
        reasoning_mask_list.append(rmask[:max_len])
        attention_mask_list.append(attn_mask)

    return {
        "input_ids": torch.stack(input_ids_list),  # (B, T)
        "target_ids": torch.stack(target_ids_list),  # (B, T)
        "reasoning_mask": torch.stack(reasoning_mask_list),  # (B, T)
        "attention_mask": torch.stack(attention_mask_list),  # (B, T)
    }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class CoTDistillTrainer:
    """Chain-of-Thought distillation trainer (Ho et al. 2022)."""

    def __init__(
        self,
        student: nn.Module,
        teacher: nn.Module | None,
        config: CoTDistillConfig,
    ) -> None:
        self.student = student
        self.teacher = teacher  # may be None for CE-only training
        self.config = config
        self.optimizer = torch.optim.AdamW(student.parameters(), lr=config.lr)

    def train_step(self, samples: list[CoTSample]) -> dict[str, float]:
        """One training step. Returns metrics dict with 'loss', 'cot_loss', 'answer_loss'."""
        self.student.train()
        self.optimizer.zero_grad()

        batch = prepare_cot_batch(samples, max_len=self.config.max_seq_len)
        input_ids = batch["input_ids"]
        target_ids = batch["target_ids"]
        reasoning_mask = batch["reasoning_mask"]

        # Forward student — handle tuple output (loss, logits, pkv)
        out = self.student(input_ids)
        student_logits = out[1] if isinstance(out, tuple) else out  # (B, T, vocab)

        if self.config.cot_loss_type == "kl" and self.teacher is not None:
            # Get teacher logits (no grad)
            self.teacher.train(False)
            with torch.no_grad():
                t_out = self.teacher(input_ids)
                teacher_logits = t_out[1] if isinstance(t_out, tuple) else t_out

            loss = knowledge_distillation_loss(
                student_logits,
                teacher_logits,
                target_ids,
                temperature=self.config.temperature,
                alpha=self.config.alpha,
            )
            # Compute CE split for reporting metrics
            _, metrics = cot_cross_entropy_loss(
                student_logits.detach(), target_ids, reasoning_mask, alpha=self.config.alpha
            )
            metrics["loss"] = loss.item()
        else:
            # CE-only training
            loss, metrics = cot_cross_entropy_loss(
                student_logits,
                target_ids,
                reasoning_mask,
                alpha=self.config.alpha,
            )
            metrics["loss"] = metrics["total_loss"]

        loss.backward()
        self.optimizer.step()

        return metrics

    def generate_rationale(
        self,
        model: nn.Module,
        question_ids: Tensor,  # (T_q,)
        max_new_tokens: int = 16,
    ) -> Tensor:
        """Autoregressively generate a rationale continuation.

        Returns (max_new_tokens,) ids.
        """
        model.train(False)
        generated: list[Tensor] = []
        current_ids = question_ids.unsqueeze(0)  # (1, T_q)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                out = model(current_ids)
                logits = out[1] if isinstance(out, tuple) else out  # (1, T, vocab)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
                generated.append(next_token.squeeze(0))  # scalar or (1,)
                current_ids = torch.cat([current_ids, next_token], dim=1)

        return torch.stack(generated, dim=0).reshape(max_new_tokens)  # (max_new_tokens,)
