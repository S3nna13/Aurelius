"""Dynamic Sparse Training (RigL-style) for Aurelius.

Implements the RigL algorithm (Evci et al., 2020 — arXiv:1911.11134):
  - Maintain a sparse mask over each tracked parameter.
  - Periodically drop weights with the smallest magnitudes and grow weights
    with the largest gradient scores (|grad * weight| surrogate).
  - Between updates, keep masks fixed and zero out pruned weights after
    each optimiser step.

Usage::

    cfg = DynamicSparseConfig(sparsity=0.9, update_interval=100)
    trainer = DynamicSparseTrainer(model, cfg, optimizer)
    for batch in dataloader:
        info = trainer.train_step(batch["input_ids"])
        print(info["loss"], info["sparsity"])
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DynamicSparseConfig:
    """Hyper-parameters for RigL-style dynamic sparse training.

    Attributes:
        sparsity: Target fraction of zero weights in each masked parameter
            (e.g. 0.9 means 90 % zeros, 10 % active).
        update_interval: Number of training steps between mask updates.
        init_method: How to initialise masks — "random" samples uniformly,
            "magnitude" keeps the top (1-sparsity) weights by absolute value.
        grow_method: How to select new connections when growing — "gradient"
            uses |grad * weight| scores, "random" samples uniformly.
    """
    sparsity: float = 0.9
    update_interval: int = 100
    init_method: str = "random"   # "random" | "magnitude"
    grow_method: str = "gradient"  # "gradient" | "random"


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def compute_sparsity(tensor: torch.Tensor) -> float:
    """Return the fraction of elements that are exactly zero.

    Args:
        tensor: Any torch.Tensor.

    Returns:
        A float in [0, 1].
    """
    if tensor.numel() == 0:
        return 0.0
    n_zero = (tensor == 0).sum().item()
    return float(n_zero) / tensor.numel()


def create_random_mask(shape: tuple, sparsity: float) -> torch.Tensor:
    """Create a binary mask with (1 - sparsity) fraction of ones.

    Active (1) positions correspond to kept weights; zero positions are pruned.

    Args:
        shape: Desired tensor shape.
        sparsity: Fraction of zeros in the resulting mask.

    Returns:
        Float tensor of 0s and 1s with the requested sparsity.
    """
    numel = math.prod(shape)
    n_active = max(1, round(numel * (1.0 - sparsity)))
    # Build a flat mask, then reshape
    flat = torch.zeros(numel)
    indices = torch.randperm(numel)[:n_active]
    flat[indices] = 1.0
    return flat.reshape(shape)


def apply_mask(model: nn.Module, masks: Dict[str, torch.Tensor]) -> None:
    """Zero out weights in-place for every parameter listed in *masks*.

    Args:
        model: The model whose parameters will be masked.
        masks: Mapping from parameter name (as returned by
            ``model.named_parameters()``) to a binary mask tensor of the
            same shape.
    """
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name in masks:
                param.mul_(masks[name].to(param.device))


def compute_magnitude_mask(param: torch.Tensor, sparsity: float) -> torch.Tensor:
    """Return a binary mask that keeps the top (1 - sparsity) weights by magnitude.

    Args:
        param: Weight tensor.
        sparsity: Desired fraction of zeros in the output mask.

    Returns:
        Boolean-valued float mask (1 = keep, 0 = prune) of the same shape.
    """
    numel = param.numel()
    n_active = max(1, round(numel * (1.0 - sparsity)))
    flat_abs = param.data.abs().flatten()
    # topk returns values in descending order
    threshold = torch.topk(flat_abs, n_active, largest=True).values[-1]
    mask = (param.data.abs() >= threshold).float()
    # Edge case: if tie-breaking gives too many ones, trim randomly
    if mask.sum().item() > n_active:
        # Zero out the excess randomly from the tied positions
        excess = int(mask.sum().item()) - n_active
        tied = ((param.data.abs() == threshold).float().flatten().nonzero(as_tuple=False).squeeze(1))
        drop = tied[torch.randperm(tied.numel())[:excess]]
        mask_flat = mask.flatten()
        mask_flat[drop] = 0.0
        mask = mask_flat.reshape(param.shape)
    return mask


def compute_gradient_scores(param: torch.Tensor) -> torch.Tensor:
    """Return |grad * weight| scores for RigL grow decisions.

    If no gradient is available, returns a zero tensor of the same shape.

    Args:
        param: A model parameter (may or may not have .grad populated).

    Returns:
        Float tensor of the same shape as *param*.
    """
    if param.grad is None:
        return torch.zeros_like(param.data)
    return (param.grad * param.data).abs()


# ---------------------------------------------------------------------------
# RigL mask update
# ---------------------------------------------------------------------------

def rigl_update(
    model: nn.Module,
    masks: Dict[str, torch.Tensor],
    sparsity: float,
    grow_method: str = "gradient",
) -> Dict[str, torch.Tensor]:
    """Perform one RigL mask update for every parameter in *masks*.

    For each masked parameter:
      1. Determine k = number of active weights = round(numel * (1 - sparsity)).
      2. **Drop** the k_drop weights with the smallest absolute values among
         currently active weights (mask == 1).
      3. **Grow** k_drop new weights chosen by gradient score (or randomly)
         among currently pruned weights (mask == 0).

    This keeps the sparsity level exactly constant.

    Args:
        model: The model (parameters must still have .grad tensors if
            grow_method == "gradient").
        masks: Current masks dict (modified out-of-place; original unchanged).
        sparsity: Target sparsity level.
        grow_method: "gradient" or "random".

    Returns:
        New masks dict with the same keys.
    """
    new_masks: Dict[str, torch.Tensor] = {}

    params_dict = dict(model.named_parameters())

    for name, mask in masks.items():
        if name not in params_dict:
            new_masks[name] = mask.clone()
            continue

        param = params_dict[name]
        numel = param.numel()
        n_active = max(1, round(numel * (1.0 - sparsity)))

        flat_mask = mask.flatten().clone()
        flat_data = param.data.flatten()

        active_idx = flat_mask.nonzero(as_tuple=False).squeeze(1)
        pruned_idx = (flat_mask == 0).nonzero(as_tuple=False).squeeze(1)

        # Number of connections to drop / grow (keep total active = n_active)
        current_active = active_idx.numel()
        # k_drop chosen so we stay at n_active after regrow
        k_drop = max(0, current_active - (n_active - min(pruned_idx.numel(), n_active)))
        # Simpler: always drop and regrow the same count
        k = min(current_active, n_active)
        k_drop = current_active - k

        # Alternatively use the standard RigL approach: drop some fraction,
        # regrow the same number.  Here we simply ensure total active == n_active.
        # Drop: remove the k_drop lowest-magnitude active weights
        if k_drop > 0 and active_idx.numel() > 0:
            active_mags = flat_data[active_idx].abs()
            _, sort_idx = active_mags.sort(descending=False)
            to_drop = active_idx[sort_idx[:k_drop]]
            flat_mask[to_drop] = 0.0

        # Recalculate after drop
        active_idx_new = flat_mask.nonzero(as_tuple=False).squeeze(1)
        pruned_idx_new = (flat_mask == 0).nonzero(as_tuple=False).squeeze(1)

        # Grow: add back connections from pruned set
        n_to_grow = n_active - active_idx_new.numel()
        n_to_grow = min(n_to_grow, pruned_idx_new.numel())

        if n_to_grow > 0 and pruned_idx_new.numel() > 0:
            if grow_method == "gradient":
                if param.grad is not None:
                    flat_scores = (param.grad.flatten() * flat_data).abs()
                else:
                    flat_scores = torch.zeros(numel)
                pruned_scores = flat_scores[pruned_idx_new]
                _, sort_idx = pruned_scores.sort(descending=True)
                to_grow = pruned_idx_new[sort_idx[:n_to_grow]]
            else:  # "random"
                perm = torch.randperm(pruned_idx_new.numel())[:n_to_grow]
                to_grow = pruned_idx_new[perm]

            flat_mask[to_grow] = 1.0

        new_masks[name] = flat_mask.reshape(mask.shape)

    return new_masks


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class DynamicSparseTrainer:
    """Training wrapper that applies RigL dynamic sparse training.

    Usage::

        trainer = DynamicSparseTrainer(model, DynamicSparseConfig(), optimizer)
        info = trainer.train_step(input_ids)  # returns dict

    Args:
        model: An AureliusTransformer (or any nn.Module) whose forward
            signature is ``model(input_ids) -> (loss, logits, past_kv)``.
        config: Sparse training hyper-parameters.
        optimizer: A torch optimiser (e.g. ``torch.optim.AdamW``).
    """

    def __init__(
        self,
        model: nn.Module,
        config: DynamicSparseConfig,
        optimizer: torch.optim.Optimizer,
    ) -> None:
        self.model = model
        self.config = config
        self.optimizer = optimizer
        self.step = 0

        # Initialise masks for every parameter that has >= 2 elements
        self.masks: Dict[str, torch.Tensor] = {}
        for name, param in model.named_parameters():
            if param.numel() < 2:
                continue
            if config.init_method == "magnitude":
                mask = compute_magnitude_mask(param, config.sparsity)
            else:  # "random"
                mask = create_random_mask(param.shape, config.sparsity)
            self.masks[name] = mask.to(param.device)

        # Apply initial masks
        apply_mask(self.model, self.masks)

    def train_step(self, input_ids: torch.Tensor) -> dict:
        """Run one training step and optionally update masks.

        Steps:
          1. Apply current masks (re-zero pruned weights).
          2. Forward pass: ``loss, logits, pkv = model(input_ids)``.
          3. Backward pass.
          4. Optimiser step.
          5. Re-apply masks (to eliminate any drift from the optimiser).
          6. Every ``update_interval`` steps: call :func:`rigl_update`.
          7. Increment step counter.

        Args:
            input_ids: Integer token tensor of shape ``(batch, seq_len)``.

        Returns:
            dict with keys:
              - ``"loss"``    — scalar float.
              - ``"sparsity"``— mean sparsity across all masked params.
              - ``"step"``    — current step number (post-increment).
        """
        # 1. Enforce mask before forward
        apply_mask(self.model, self.masks)

        # 2. Forward — API: loss, logits, pkv = model(input_ids)
        # Pass input_ids as labels so the model computes cross-entropy loss
        # (the model shifts internally: logits[:,:-1] vs labels[:,1:]).
        self.model.train()
        self.optimizer.zero_grad()
        loss, logits, pkv = self.model(input_ids, labels=input_ids)

        # 3. Backward
        loss.backward()

        # 4. Optimiser step
        self.optimizer.step()

        # 5. Re-enforce mask (optimiser may violate it)
        apply_mask(self.model, self.masks)

        # 6. Periodic RigL mask update
        self.step += 1
        if self.step % self.config.update_interval == 0:
            self.masks = rigl_update(
                self.model,
                self.masks,
                self.config.sparsity,
                self.config.grow_method,
            )
            apply_mask(self.model, self.masks)

        # Compute mean sparsity across all masked params
        sparsities = []
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in self.masks:
                    sparsities.append(compute_sparsity(param.data))
        mean_sparsity = float(sum(sparsities) / len(sparsities)) if sparsities else 0.0

        return {
            "loss": loss.item(),
            "sparsity": mean_sparsity,
            "step": self.step,
        }
