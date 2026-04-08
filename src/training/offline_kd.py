"""Offline knowledge distillation from pre-computed teacher logits.

Teacher logits are pre-saved as .npy files with shape (n_tokens, vocab_size).
Student trains to match these distributions (no teacher in memory required).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


@dataclass
class OfflineKDConfig:
    temperature: float = 4.0    # KD temperature
    alpha: float = 0.5          # mix: alpha * KD_loss + (1-alpha) * CE_loss
    top_k_logits: int = 0       # if > 0, only keep top-k teacher logits (sparse KD)
    # 0 means use all vocab


class TeacherLogitDataset(Dataset):
    """Dataset that serves (input_ids, labels, teacher_logits) windows.

    Expects:
        input_ids_path: path to .npy file, dtype uint16, shape (N,)
        teacher_logits_path: path to .npy file, dtype float16, shape (N, vocab_size)

    Each item is a sliding window of length seq_len.
    Labels are input_ids shifted right by 1 (next-token prediction).
    Teacher logits window aligns with input_ids window (position i predicts i+1).
    """

    def __init__(
        self,
        input_ids_path: str,
        teacher_logits_path: str,
        seq_len: int,
    ) -> None:
        self.seq_len = seq_len
        # Memory-mapped loading: avoids loading full arrays into RAM
        self.input_ids = np.load(input_ids_path, mmap_mode="r")
        self.teacher_logits = np.load(teacher_logits_path, mmap_mode="r")

        assert self.input_ids.ndim == 1, "input_ids must be 1-D"
        assert self.teacher_logits.ndim == 2, "teacher_logits must be 2-D (N, vocab)"
        assert len(self.input_ids) == len(self.teacher_logits), (
            "input_ids and teacher_logits must have the same first dimension"
        )

        # Number of full windows: need seq_len tokens + 1 label token
        self._n_windows = max(0, len(self.input_ids) - seq_len)

    def __len__(self) -> int:
        return self._n_windows

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        i = idx
        # input window: [i, i+seq_len)
        input_ids = torch.from_numpy(
            self.input_ids[i : i + self.seq_len].astype(np.int64)
        )
        # labels: next-token for each position → [i+1, i+seq_len+1)
        labels = torch.from_numpy(
            self.input_ids[i + 1 : i + self.seq_len + 1].astype(np.int64)
        )
        # teacher logits for the input window (cast float16 → float32)
        teacher_logits = torch.from_numpy(
            self.teacher_logits[i : i + self.seq_len].astype(np.float32)
        )
        return {
            "input_ids": input_ids,
            "labels": labels,
            "teacher_logits": teacher_logits,
        }


def offline_kd_loss(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    labels: torch.Tensor,
    cfg: OfflineKDConfig,
) -> torch.Tensor:
    """Combined KD + CE loss using pre-computed teacher logits.

    Args:
        student_logits: (B, S, V) — raw student output logits.
        teacher_logits: (B, S, V) — pre-computed teacher logits (float32).
        labels: (B, S) — target token ids for CE loss.
        cfg: OfflineKDConfig with temperature, alpha, top_k_logits.

    Returns:
        Scalar loss tensor: alpha * kd_loss + (1 - alpha) * ce_loss.
    """
    B, S, V = student_logits.shape
    T = cfg.temperature

    # --- CE loss ---
    ce_loss = F.cross_entropy(
        student_logits.contiguous().view(-1, V),
        labels.contiguous().view(-1),
    )

    # --- Sparse KD: zero out teacher logits outside top-k ---
    if cfg.top_k_logits > 0:
        k = min(cfg.top_k_logits, V)
        # Find top-k values per position
        topk_vals, topk_idx = torch.topk(teacher_logits, k, dim=-1)  # (B, S, k)
        # Build sparse teacher logits: fill with large negative (soft-zero after softmax)
        sparse = torch.full_like(teacher_logits, float("-inf"))
        sparse.scatter_(-1, topk_idx, topk_vals)
        teacher_logits = sparse

    # --- KD loss: KL(student_soft || teacher_soft) * T^2 ---
    student_log_soft = F.log_softmax(student_logits / T, dim=-1)   # (B, S, V)
    teacher_soft = F.softmax(teacher_logits / T, dim=-1)            # (B, S, V)

    kd_loss = F.kl_div(
        student_log_soft.view(-1, V),
        teacher_soft.view(-1, V),
        reduction="batchmean",
    ) * (T ** 2)

    total = cfg.alpha * kd_loss + (1.0 - cfg.alpha) * ce_loss
    return total


class OfflineKDTrainer:
    """Trains a student model using offline knowledge distillation.

    No teacher model is needed at runtime — teacher logits are pre-loaded
    from disk via TeacherLogitDataset.

    Args:
        model: Student model (nn.Module). Forward signature must accept
               input_ids and return (_, logits, _) or just logits.
        cfg: OfflineKDConfig.
        lr: Learning rate for AdamW.
    """

    def __init__(
        self,
        model: "torch.nn.Module",
        cfg: OfflineKDConfig | None = None,
        lr: float = 1e-4,
    ) -> None:
        self.model = model
        self.cfg = cfg or OfflineKDConfig()
        self.optimizer = torch.optim.AdamW(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
        )

    def train_step(self, batch: dict) -> dict[str, float]:
        """One offline KD training step.

        Args:
            batch: dict with keys "input_ids", "labels", "teacher_logits".
                   All tensors with batch dimension.

        Returns:
            {"loss": float, "kd_loss": float, "ce_loss": float}
        """
        self.model.train()
        self.optimizer.zero_grad()

        input_ids = batch["input_ids"]            # (B, S)
        labels = batch["labels"]                   # (B, S)
        teacher_logits = batch["teacher_logits"]  # (B, S, V)

        # Forward pass — support both (loss, logits, aux) and plain logits returns
        output = self.model(input_ids)
        if isinstance(output, tuple):
            student_logits = output[1]
        else:
            student_logits = output  # (B, S, V)

        B, S, V = student_logits.shape
        T = self.cfg.temperature

        # Compute individual losses for logging
        ce_loss = F.cross_entropy(
            student_logits.contiguous().view(-1, V),
            labels.contiguous().view(-1),
        )

        tl = teacher_logits
        if self.cfg.top_k_logits > 0:
            k = min(self.cfg.top_k_logits, V)
            topk_vals, topk_idx = torch.topk(tl, k, dim=-1)
            sparse = torch.full_like(tl, float("-inf"))
            sparse.scatter_(-1, topk_idx, topk_vals)
            tl = sparse

        student_log_soft = F.log_softmax(student_logits / T, dim=-1)
        teacher_soft = F.softmax(tl / T, dim=-1)
        kd_loss = F.kl_div(
            student_log_soft.view(-1, V),
            teacher_soft.view(-1, V),
            reduction="batchmean",
        ) * (T ** 2)

        total_loss = self.cfg.alpha * kd_loss + (1.0 - self.cfg.alpha) * ce_loss

        total_loss.backward()
        self.optimizer.step()

        return {
            "loss": total_loss.item(),
            "kd_loss": kd_loss.item(),
            "ce_loss": ce_loss.item(),
        }
