# Token Superposition Training (TST) — Implementation Guide
# Based on: arXiv 2605.06546 (Nous Research, May 2026)

"""
TST: 2.5× faster pretraining via bagged tokens + multi-hot CE loss.
- Works by training on averaged token bags instead of individual tokens
- Only during Phase 1 (first r fraction of steps)
- Zero inference overhead — final model identical to baseline
- Validated: 270M → 10B models (2.2–2.7× speedup)

When to use:
  ✅ Large-scale pretraining (>30B tokens)
  ✅ Continued-pretraining with massive new corpus
  ❌ NOT for fine-tuning/SFT (small datasets don't benefit)

Bag size selection:
  Model size      bag_size (s)    phase_ratio (r)
  ------------    ------------    ---------------
  < 1B            4               0.25
  1–3B            6               0.30
  3–10B           8–12            0.30–0.35
  10B–70B         12–16           0.35–0.40
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class TokenSuperpositionTrainer:
    """Two-phase trainer: superposition → standard NTP."""

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        bag_size: int = 6,
        phase_ratio: float = 0.30,
        use_power_law: bool = False,
        device: str = "cuda",
    ):
        self.model = model
        self.optimizer = optimizer
        self.bag_size = bag_size
        self.phase_ratio = phase_ratio
        self.use_power_law = use_power_law
        self.device = device

    def is_phase1(self, step: int, total_steps: int) -> bool:
        """Return True if we're in the superposition phase."""
        return step < total_steps * self.phase_ratio

    def superposition_transform(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Convert [batch, seq] → [batch, num_bags, bag_size].

        Example: seq=1024, bag_size=8 → num_bags=128
        """
        bs, seq_len = input_ids.shape
        if seq_len % self.bag_size != 0:
            raise ValueError(
                f"seq_len ({seq_len}) must be divisible by bag_size ({self.bag_size})"
            )
        bags = input_ids.view(bs, seq_len // self.bag_size, self.bag_size)
        return bags  # shape: [batch, num_bags, bag_size]

    def multi_hot_cross_entropy(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Multi-hot cross-entropy loss.

        Args:
            logits: [batch, num_bags, vocab] — prediction per bag
            labels: [batch, num_bags, bag_size] — each position is a token ID

        Returns:
            Scalar loss
        """
        bs, num_bags, vocab = logits.shape
        _, _, bag_size = labels.shape

        loss = 0.0
        weights = []

        # Power-law weighting: later tokens in bag get less weight
        for i in range(bag_size):
            w = 1.0 / (i + 1) if self.use_power_law else 1.0
            weights.append(w)

        total_w = sum(weights)

        for i, w in enumerate(weights):
            # Extract i-th token from each bag as target
            target_i = labels[:, :, i]  # [batch, num_bags]

            # Compute CE: logits flatten to [batch*num_bags, vocab]
            loss_i = F.cross_entropy(
                logits.flatten(0, 1),
                target_i.flatten(0, 1),
            )
            loss += (w / total_w) * loss_i

        return loss / bag_size

    def train_step(self, batch: dict, step: int, total_steps: int) -> torch.Tensor:
        """
        Single training step (forward + backward).

        Args:
            batch: {"input_ids": [batch, seq_len]}
            step: current step index
            total_steps: total training steps

        Returns:
            scalar loss
        """
        input_ids = batch["input_ids"].to(self.device)

        if self.is_phase1(step, total_steps):
            # PHASE 1: Superposition
            bagged_ids = self.superposition_transform(input_ids)
            # Model forward expects embeddings; implement averaging in model's embedding layer
            logits = self.model(bagged_ids, mode="superposition")  # [batch, num_bags, vocab]

            # Labels = next bag (shift left by 1)
            labels = torch.roll(bagged_ids, shifts=-1, dims=1)
            # Discard last bag (no next bag)
            logits = logits[:, :-1, :]
            labels = labels[:, :-1, :]

            loss = self.multi_hot_cross_entropy(logits, labels)

        else:
            # PHASE 2: Standard next-token prediction
            loss = self.model(input_ids, mode="standard")  # standard CE

        # Backward + optimizer step
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss


def create_superposition_dataloader(
    base_dataset,
    bag_size: int,
    phase: str = "superposition",
    batch_size: int = 32,
):
    """
    Wrap an existing dataset to output bagged format.

    Args:
        base_dataset: yields {"input_ids": torch.Tensor [seq_len]}
        bag_size: number of tokens per bag
        phase: "superposition" (bagged) or "recovery" (standard)
        batch_size: batch size for dataloader

    Returns:
        DataLoader yielding bagged or standard batches
    """
    from torch.utils.data import Dataset

    class SuperpositionDataset(Dataset):
        def __init__(self, base, bag_size, phase):
            self.base = base
            self.bag_size = bag_size
            self.phase = phase

        def __len__(self):
            return len(self.base)

        def __getitem__(self, idx):
            item = self.base[idx]
            input_ids = item["input_ids"]

            if self.phase == "superposition":
                # Pad to multiple of bag_size
                pad_len = (self.bag_size - len(input_ids) % self.bag_size) % self.bag_size
                if pad_len > 0:
                    input_ids = F.pad(input_ids, (0, pad_len))

                # Reshape into bags
                num_bags = len(input_ids) // self.bag_size
                bags = input_ids.view(num_bags, self.bag_size)
                return {"bagged_ids": bags}
            else:
                # Recovery phase: standard format
                return item

    dataset = SuperpositionDataset(base_dataset, bag_size, phase)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)


# This module intentionally has no executable demo block.  Use
# `TokenSuperpositionTrainer` and `create_superposition_dataloader` from a
# project-specific training script that supplies a real model and dataset.
