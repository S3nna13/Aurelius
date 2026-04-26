"""Scheduled sampling to reduce exposure bias during training."""

import math

import torch
import torch.nn.functional as F
from torch import LongTensor, Tensor


class ScheduleType:
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    SIGMOID = "sigmoid"


class ScheduledSampler:
    def __init__(self, schedule: str = "linear", k: float = 1.0):
        self.schedule = schedule
        self.k = k

    def teacher_forcing_prob(self, step: int, total_steps: int) -> float:
        if total_steps <= 0:
            return 1.0

        if self.schedule == ScheduleType.LINEAR:
            return max(0.0, 1.0 - step / total_steps)

        elif self.schedule == ScheduleType.EXPONENTIAL:
            return float(self.k**step)

        elif self.schedule == ScheduleType.SIGMOID:
            return float(self.k / (self.k + math.exp(step / 1000.0)))

        return 1.0

    def sample_input(
        self,
        ground_truth_ids: LongTensor,
        model_logits: Tensor,
        step: int,
        total_steps: int,
    ) -> LongTensor:
        p = self.teacher_forcing_prob(step, total_steps)
        model_tokens = model_logits.argmax(dim=-1)
        mask = torch.bernoulli(torch.full(ground_truth_ids.shape, p, dtype=torch.float32)).bool()
        result = torch.where(mask, ground_truth_ids, model_tokens)
        return result.long()

    def mixing_loss(
        self,
        model_logits: Tensor,
        targets: LongTensor,
        step: int,
        total_steps: int,
    ) -> Tensor:
        p = self.teacher_forcing_prob(step, total_steps)
        ce = F.cross_entropy(
            model_logits.view(-1, model_logits.size(-1)),
            targets.view(-1),
        )
        return ce * p
