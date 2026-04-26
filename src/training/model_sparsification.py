"""
Model Sparsification: magnitude pruning, movement pruning, structured head pruning,
LoRA regrowth, and mask scheduling for the Aurelius LLM research project.

Pure native PyTorch — no transformers, einops, or other heavy dependencies.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn

# ---------------------------------------------------------------------------
# SparsificationConfig
# ---------------------------------------------------------------------------


@dataclass
class SparsificationConfig:
    """Configuration for the full sparsification pipeline."""

    target_sparsity: float = 0.9
    initial_sparsity: float = 0.0
    begin_step: int = 0
    end_step: int = 500
    prune_freq: int = 50
    target_head_sparsity: float = 0.5
    rank: int = 4
    lr: float = 1e-4


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prunable_named_params(model: nn.Module) -> list[tuple[str, nn.Parameter]]:
    """Return (name, param) pairs for weight tensors with >= 2 dimensions."""
    return [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad and param.dim() >= 2
    ]


# ---------------------------------------------------------------------------
# MagnitudePruner
# ---------------------------------------------------------------------------


class MagnitudePruner:
    """Unstructured magnitude-based weight pruning."""

    def __init__(self, model: nn.Module, target_sparsity: float = 0.9) -> None:
        self.target_sparsity = target_sparsity

    # ------------------------------------------------------------------
    def compute_thresholds(self, model: nn.Module) -> dict[str, float]:
        """Per-layer magnitude threshold achieving target_sparsity."""
        thresholds: dict[str, float] = {}
        for name, param in _prunable_named_params(model):
            magnitudes = param.data.abs().flatten()
            k = int(math.floor(self.target_sparsity * magnitudes.numel()))
            k = max(0, min(k, magnitudes.numel() - 1))
            if k == 0:
                thresholds[name] = 0.0
            else:
                sorted_mags, _ = torch.sort(magnitudes)
                thresholds[name] = sorted_mags[k - 1].item()
        return thresholds

    # ------------------------------------------------------------------
    def create_masks(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """Binary masks: 1 where |weight| > threshold (keep), 0 elsewhere (prune)."""
        thresholds = self.compute_thresholds(model)
        masks: dict[str, torch.Tensor] = {}
        for name, param in _prunable_named_params(model):
            thr = thresholds[name]
            masks[name] = (param.data.abs() > thr).float()
        return masks

    # ------------------------------------------------------------------
    def apply_masks(self, model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
        """Zero out pruned weights in-place (weight ← weight * mask)."""
        param_dict = dict(model.named_parameters())
        for name, mask in masks.items():
            if name in param_dict:
                with torch.no_grad():
                    param_dict[name].data.mul_(mask)

    # ------------------------------------------------------------------
    def actual_sparsity(self, model: nn.Module) -> float:
        """Fraction of zero parameters across all prunable layers."""
        total = 0
        zeros = 0
        for _, param in _prunable_named_params(model):
            total += param.numel()
            zeros += (param.data == 0).sum().item()
        if total == 0:
            return 0.0
        return zeros / total


# ---------------------------------------------------------------------------
# MovementPruner
# ---------------------------------------------------------------------------


class MovementPruner:
    """Movement pruning: importance = |weight × gradient|, accumulated over steps."""

    def __init__(
        self,
        model: nn.Module,
        target_sparsity: float = 0.9,
        warmup_steps: int = 100,
    ) -> None:
        self.target_sparsity = target_sparsity
        self.warmup_steps = warmup_steps
        # Initialise score accumulators to zero
        self.scores: dict[str, torch.Tensor] = {
            name: torch.zeros_like(param.data) for name, param in _prunable_named_params(model)
        }
        self._step = 0

    # ------------------------------------------------------------------
    def update_scores(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,  # noqa: F841 — kept for API compatibility
    ) -> None:
        """Accumulate movement scores: |weight * grad|."""
        for name, param in _prunable_named_params(model):
            if param.grad is not None and name in self.scores:
                movement = (param.data * param.grad).abs()
                self.scores[name] = self.scores[name] + movement
        self._step += 1

    # ------------------------------------------------------------------
    def create_masks(self, threshold_percentile: float) -> dict[str, torch.Tensor]:
        """Binary masks based on accumulated movement scores at given percentile."""
        masks: dict[str, torch.Tensor] = {}
        for name, score in self.scores.items():
            flat = score.flatten()
            if flat.numel() == 0:
                masks[name] = torch.ones_like(score)
                continue
            thr_idx = int(math.floor(threshold_percentile / 100.0 * flat.numel()))
            thr_idx = max(0, min(thr_idx, flat.numel() - 1))
            sorted_scores, _ = torch.sort(flat)
            thr = sorted_scores[thr_idx].item()
            masks[name] = (score > thr).float()
        return masks

    # ------------------------------------------------------------------
    def apply_masks(self, model: nn.Module, masks: dict[str, torch.Tensor]) -> None:
        """Zero out pruned weights in-place."""
        param_dict = dict(model.named_parameters())
        for name, mask in masks.items():
            if name in param_dict:
                with torch.no_grad():
                    param_dict[name].data.mul_(mask)


# ---------------------------------------------------------------------------
# SparsityScheduler
# ---------------------------------------------------------------------------


class SparsityScheduler:
    """Cubic sparsity schedule from initial_sparsity → final_sparsity."""

    def __init__(
        self,
        initial_sparsity: float = 0.0,
        final_sparsity: float = 0.9,
        begin_step: int = 0,
        end_step: int = 1000,
    ) -> None:
        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.begin_step = begin_step
        self.end_step = end_step

    # ------------------------------------------------------------------
    def current_sparsity(self, step: int) -> float:
        """
        Cubic interpolation:
            s_t = s_f + (s_0 - s_f) * (1 - (t - t_0) / (t_1 - t_0))^3
        Clamped to [initial_sparsity, final_sparsity].
        """
        s0 = self.initial_sparsity
        sf = self.final_sparsity
        t0 = self.begin_step
        t1 = self.end_step

        if step <= t0:
            return s0
        if step >= t1:
            return sf

        frac = (step - t0) / (t1 - t0)
        sparsity = sf + (s0 - sf) * (1.0 - frac) ** 3
        # Clamp to valid range (handles floating-point drift)
        lo, hi = min(s0, sf), max(s0, sf)
        return max(lo, min(hi, sparsity))

    # ------------------------------------------------------------------
    def should_prune(self, step: int, freq: int = 100) -> bool:
        """True when step is a multiple of freq and within [begin_step, end_step]."""
        return self.begin_step <= step <= self.end_step and freq > 0 and step % freq == 0


# ---------------------------------------------------------------------------
# HeadPruner
# ---------------------------------------------------------------------------


class HeadPruner:
    """Structured attention-head pruning based on attention entropy."""

    def __init__(self, model: nn.Module, target_head_sparsity: float = 0.5) -> None:
        self.model = model
        self.target_head_sparsity = target_head_sparsity

    # ------------------------------------------------------------------
    @staticmethod
    def _entropy(attn: torch.Tensor) -> torch.Tensor:
        """Shannon entropy of attention distributions, shape [..., seq, seq] → [...]."""
        # Clamp to avoid log(0)
        p = attn.clamp(min=1e-9)
        return -(p * p.log()).sum(dim=-1).mean(dim=-1)  # mean over query positions

    # ------------------------------------------------------------------
    def compute_head_importance(self, attn_weights: list[torch.Tensor]) -> dict[str, torch.Tensor]:
        """
        Importance = mean |attention entropy| per head.

        Parameters
        ----------
        attn_weights : list of Tensor, each [B, n_heads, T, T]

        Returns
        -------
        dict mapping layer index string → importance Tensor [n_heads]
        """
        importance: dict[str, torch.Tensor] = {}
        for layer_idx, attn in enumerate(attn_weights):
            # attn: [B, n_heads, T, T]
            # entropy per head: [B, n_heads]
            ent = self._entropy(attn)  # [B, n_heads]
            importance[f"layer_{layer_idx}"] = ent.abs().mean(dim=0)  # [n_heads]
        return importance

    # ------------------------------------------------------------------
    def prune_heads(
        self, importance: dict[str, torch.Tensor], n_to_prune: int
    ) -> dict[str, list[int]]:
        """
        Select the n_to_prune heads with lowest importance globally.

        Returns
        -------
        dict mapping layer_name → list of head indices to prune
        """
        # Flatten all (layer_name, head_idx, importance_value) triples
        candidates: list[tuple[str, int, float]] = []
        for layer_name, imp in importance.items():
            for head_idx in range(imp.numel()):
                candidates.append((layer_name, head_idx, imp[head_idx].item()))

        # Sort ascending (lowest importance first)
        candidates.sort(key=lambda x: x[2])

        to_prune: dict[str, list[int]] = {}
        for layer_name, head_idx, _ in candidates[:n_to_prune]:
            to_prune.setdefault(layer_name, []).append(head_idx)
        return to_prune

    # ------------------------------------------------------------------
    def apply_head_masks(self, model: nn.Module, head_masks: dict[str, list[int]]) -> None:
        """
        Zero out attention projection weights for pruned heads.

        Heuristic: looks for sub-modules whose name contains a layer key
        and applies zeroing to any weight whose leading dimension corresponds
        to a head index.  Works with standard nn.MultiheadAttention and
        custom attention layers that expose an `out_proj` weight.
        """
        dict(model.named_modules())
        named_params = dict(model.named_parameters())

        for layer_key, head_indices in head_masks.items():
            # Find all parameters that belong to this layer key
            for param_name, param in named_params.items():
                if layer_key in param_name and param.dim() >= 2:
                    with torch.no_grad():
                        for h in head_indices:
                            if h < param.shape[0]:
                                param.data[h].zero_()


# ---------------------------------------------------------------------------
# LoRARegrowth
# ---------------------------------------------------------------------------


class LoRARegrowth:
    """Re-grow pruned capacity via LoRA-style low-rank perturbations."""

    def __init__(self, model: nn.Module, rank: int = 4) -> None:
        self.model = model
        self.rank = rank

    # ------------------------------------------------------------------
    def regrow_pruned(
        self, masks: dict[str, torch.Tensor]
    ) -> dict[str, tuple[nn.Parameter, nn.Parameter]]:
        """
        For each pruned weight tensor, create LoRA A/B matrices.

        Returns
        -------
        dict mapping param_name → (A, B) where delta_W = B @ A / rank
            A : [rank, d_in]
            B : [d_out, rank]
        """
        param_dict = dict(self.model.named_parameters())
        lora_params: dict[str, tuple[nn.Parameter, nn.Parameter]] = {}

        for name, mask in masks.items():
            if name not in param_dict:
                continue
            weight = param_dict[name]
            if weight.dim() < 2:
                continue

            d_out, d_in = weight.shape[0], weight.shape[1:].numel()
            r = min(self.rank, d_in, d_out)

            A = nn.Parameter(torch.randn(r, d_in, device=weight.device) * 0.01)
            B = nn.Parameter(torch.zeros(d_out, r, device=weight.device))
            lora_params[name] = (A, B)

        return lora_params

    # ------------------------------------------------------------------
    def merge_and_reprune(
        self,
        model: nn.Module,
        lora_params: dict[str, tuple[nn.Parameter, nn.Parameter]],
        masks: dict[str, torch.Tensor],
    ) -> None:
        """
        Merge LoRA deltas back into weights and re-apply masks.

        delta_W = B @ A / rank  (reshaped to match weight shape)
        """
        param_dict = dict(model.named_parameters())
        for name, (A, B) in lora_params.items():
            if name not in param_dict:
                continue
            weight = param_dict[name]
            r = A.shape[0]
            delta = (B @ A) / r  # [d_out, d_in]
            delta = delta.view_as(weight)
            with torch.no_grad():
                weight.data.add_(delta)

        # Re-apply masks
        for name, mask in masks.items():
            if name in param_dict:
                with torch.no_grad():
                    param_dict[name].data.mul_(mask)


# ---------------------------------------------------------------------------
# PruningTrainer
# ---------------------------------------------------------------------------


class PruningTrainer:
    """Thin training wrapper that integrates pruning into the forward/backward pass."""

    def __init__(
        self,
        model: nn.Module,
        pruner,
        scheduler: SparsityScheduler,
        lr: float = 1e-4,
    ) -> None:
        self.model = model
        self.pruner = pruner
        self.scheduler = scheduler
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self._current_masks: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    def train_step(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor,
        step: int,
    ) -> tuple[float, float]:
        """
        Single training step with optional pruning.

        Parameters
        ----------
        input_ids : [B, T]  long
        labels    : [B, T]  long
        step      : current global step

        Returns
        -------
        (loss_value, current_sparsity)
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Forward pass — model must accept (input_ids) and return logits [B, T, vocab]
        logits = self.model(input_ids)
        B, T, V = logits.shape
        loss = nn.functional.cross_entropy(logits.reshape(B * T, V), labels.reshape(B * T))
        loss.backward()

        # Movement pruner: update scores from gradients before optimizer step
        if isinstance(self.pruner, MovementPruner):
            self.pruner.update_scores(self.model, self.optimizer)

        self.optimizer.step()

        # Pruning step
        if self.scheduler.should_prune(step):
            sparsity = self.scheduler.current_sparsity(step)
            if isinstance(self.pruner, MagnitudePruner):
                self.pruner.target_sparsity = sparsity
                self._current_masks = self.pruner.create_masks(self.model)
            elif isinstance(self.pruner, MovementPruner):
                pct = sparsity * 100.0
                self._current_masks = self.pruner.create_masks(pct)
            self.pruner.apply_masks(self.model, self._current_masks)

        actual = self._compute_sparsity()
        return loss.item(), actual

    # ------------------------------------------------------------------
    def _compute_sparsity(self) -> float:
        total = 0
        zeros = 0
        for _, param in _prunable_named_params(self.model):
            total += param.numel()
            zeros += (param.data == 0).sum().item()
        return zeros / total if total > 0 else 0.0

    # ------------------------------------------------------------------
    def get_pruning_stats(self) -> dict[str, object]:
        total = sum(p.numel() for _, p in _prunable_named_params(self.model))
        zeros = sum((p.data == 0).sum().item() for _, p in _prunable_named_params(self.model))
        return {
            "current_sparsity": zeros / total if total > 0 else 0.0,
            "n_pruned_params": int(zeros),
            "n_total_params": int(total),
        }
