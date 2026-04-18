"""
Auxiliary Task / Multi-Task Learning
Shared backbone with task-specific heads, gradient surgery, and dynamic task weighting.
"""

import math
import copy
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# TaskHead
# ---------------------------------------------------------------------------

class TaskHead(nn.Module):
    """Task-specific output head with associated loss computation."""

    def __init__(self, d_model: int, output_size: int, task_type: str) -> None:
        super().__init__()
        if task_type not in ("lm", "classification", "regression"):
            raise ValueError(f"Unknown task_type: {task_type}")
        self.task_type = task_type
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, output_size),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)

    def compute_loss(self, features: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        features: [B, d_model] or [B, T, d_model]
        targets:  [B] for classification/lm, or [B] / [B, T] for regression
        Returns scalar loss.
        """
        logits = self.head(features)

        if self.task_type in ("lm", "classification"):
            # logits: [B, C] or [B, T, C]; targets: [B] or [B, T]
            if logits.dim() == 3:
                # [B, T, C] -> [B*T, C], targets [B, T] -> [B*T]
                B, T, C = logits.shape
                loss = F.cross_entropy(logits.reshape(B * T, C), targets.reshape(B * T))
            else:
                loss = F.cross_entropy(logits, targets)
        else:  # regression
            loss = F.mse_loss(logits.squeeze(-1), targets.float())

        return loss


# ---------------------------------------------------------------------------
# SharedBackbone
# ---------------------------------------------------------------------------

class _TransformerBlock(nn.Module):
    """Minimal transformer block (no external deps)."""

    def __init__(self, d_model: int, n_heads: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with causal mask
        T = x.size(1)
        causal_mask = torch.triu(
            torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1
        )
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, attn_mask=causal_mask, is_causal=False)
        x = x + attn_out
        x = x + self.ff(self.norm2(x))
        return x


class SharedBackbone(nn.Module):
    """Embedding + transformer blocks shared across all tasks."""

    def __init__(self, d_model: int, vocab_size: int, n_layers: int = 2) -> None:
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Simple learned positional embedding (up to 512 positions)
        self.pos_embedding = nn.Embedding(512, d_model)
        self.blocks = nn.ModuleList([
            _TransformerBlock(d_model) for _ in range(n_layers)
        ])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        input_ids: [B, T]
        Returns:
            token_repr: [B, T, d_model]  — for sequence tasks (e.g. LM)
            pooled:     [B, d_model]     — mean-pooled for classification
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0).expand(B, T)
        x = self.embedding(input_ids) + self.pos_embedding(positions)
        for block in self.blocks:
            x = block(x)
        token_repr = self.norm(x)            # [B, T, d_model]
        pooled = token_repr.mean(dim=1)      # [B, d_model]
        return token_repr, pooled


# ---------------------------------------------------------------------------
# MTLModel
# ---------------------------------------------------------------------------

class MTLModel(nn.Module):
    """Multi-task model: shared backbone + per-task heads."""

    def __init__(
        self,
        backbone: SharedBackbone,
        task_heads: Dict[str, TaskHead],
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.task_heads = nn.ModuleDict(task_heads)

    def forward(
        self, input_ids: torch.Tensor, task_name: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns (features, logits).
        features: [B, T, d_model] for 'lm', [B, d_model] for others.
        logits:   output of the task head.
        """
        token_repr, pooled = self.backbone(input_ids)
        head = self.task_heads[task_name]
        if head.task_type == "lm":
            features = token_repr          # [B, T, d_model]
        else:
            features = pooled              # [B, d_model]
        logits = head(features)
        return features, logits

    def compute_all_losses(
        self,
        input_ids: torch.Tensor,
        targets_dict: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Compute loss for each task independently."""
        token_repr, pooled = self.backbone(input_ids)
        losses: Dict[str, torch.Tensor] = {}
        for task_name, targets in targets_dict.items():
            head = self.task_heads[task_name]
            features = token_repr if head.task_type == "lm" else pooled
            losses[task_name] = head.compute_loss(features, targets)
        return losses


# ---------------------------------------------------------------------------
# GradientSurgery (PCGrad)
# ---------------------------------------------------------------------------

class GradientSurgery:
    """
    PCGrad: Project Conflicting Gradients (Yu et al., 2020).
    For each pair (i, j): if cos(g_i, g_j) < 0, project g_i onto the normal
    plane of g_j.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def compute_task_gradients(
        self,
        losses: Dict[str, torch.Tensor],
        params: List[torch.Tensor],
    ) -> Dict[str, List[torch.Tensor]]:
        """
        Compute per-task gradients w.r.t. params without accumulating.
        Returns dict task_name -> list of gradient tensors (one per param).
        """
        task_grads: Dict[str, List[torch.Tensor]] = {}
        for task_name, loss in losses.items():
            # Zero any existing grads first
            for p in params:
                if p.grad is not None:
                    p.grad = None
            grads = torch.autograd.grad(
                loss,
                params,
                retain_graph=True,
                create_graph=False,
                allow_unused=True,
            )
            # Replace None grads with zeros
            task_grads[task_name] = [
                g.clone() if g is not None else torch.zeros_like(p)
                for g, p in zip(grads, params)
            ]
        return task_grads

    @staticmethod
    def _flatten(grad_list: List[torch.Tensor]) -> torch.Tensor:
        return torch.cat([g.reshape(-1) for g in grad_list])

    @staticmethod
    def _unflatten(flat: torch.Tensor, ref_list: List[torch.Tensor]) -> List[torch.Tensor]:
        out: List[torch.Tensor] = []
        offset = 0
        for ref in ref_list:
            numel = ref.numel()
            out.append(flat[offset: offset + numel].reshape(ref.shape))
            offset += numel
        return out

    def cosine_similarity_grads(
        self, g1: List[torch.Tensor], g2: List[torch.Tensor]
    ) -> float:
        f1 = self._flatten(g1)
        f2 = self._flatten(g2)
        denom = (f1.norm() * f2.norm()).item()
        if denom < 1e-12:
            return 0.0
        return (f1 @ f2).item() / denom

    def project_conflicting(
        self, grads: Dict[str, List[torch.Tensor]]
    ) -> List[torch.Tensor]:
        """
        PCGrad: modify each task gradient by projecting away components that
        conflict with other tasks' gradients.  Returns the mean of modified
        gradients (one list of tensors for all params).
        """
        task_names = list(grads.keys())
        n_tasks = len(task_names)

        # Work in flat vector space
        flat_grads: Dict[str, torch.Tensor] = {
            t: self._flatten(grads[t]) for t in task_names
        }

        modified_flat: Dict[str, torch.Tensor] = {}
        for i, ti in enumerate(task_names):
            gi = flat_grads[ti].clone()
            for j, tj in enumerate(task_names):
                if i == j:
                    continue
                gj = flat_grads[tj]
                dot = gi @ gj
                gj_norm_sq = (gj @ gj).item()
                if dot.item() < 0.0 and gj_norm_sq > 1e-24:
                    # Project gi onto gj^perp
                    gi = gi - (dot / gj_norm_sq) * gj
            modified_flat[ti] = gi

        # Mean of modified gradients
        mean_flat = torch.stack(list(modified_flat.values())).mean(dim=0)

        # Unflatten using first task's grad list as shape reference
        ref_list = grads[task_names[0]]
        return self._unflatten(mean_flat, ref_list)


# ---------------------------------------------------------------------------
# DynamicTaskWeighter
# ---------------------------------------------------------------------------

class DynamicTaskWeighter(nn.Module):
    """
    Dynamic task weighting.

    "uncertainty": Homoscedastic uncertainty weighting (Kendall et al., 2018).
        L_total = sum_i [ 1/(2*sigma_i^2) * L_i + log(sigma_i) ]
                = sum_i [ exp(-log_sigma_i) * L_i / 2 + log_sigma_i / 2 ]
                  (using log_sigma to keep sigma > 0)
    "gradnorm":   Currently implemented as uncertainty (full GradNorm requires
                  extra infrastructure beyond a single forward call).
    "equal":      Simple mean of losses.
    """

    def __init__(self, n_tasks: int, method: str = "uncertainty") -> None:
        super().__init__()
        if method not in ("uncertainty", "gradnorm", "equal"):
            raise ValueError(f"Unknown method: {method}")
        self.method = method
        self.n_tasks = n_tasks
        # log(sigma) initialised to 0 => sigma=1 => weight=0.5
        self.log_sigma = nn.Parameter(torch.zeros(n_tasks))

    def forward(self, losses: List[torch.Tensor]) -> torch.Tensor:
        """Returns a scalar weighted total loss."""
        if len(losses) != self.n_tasks:
            raise ValueError(
                f"Expected {self.n_tasks} losses, got {len(losses)}"
            )
        stacked = torch.stack(losses)   # [n_tasks]

        if self.method == "equal":
            return stacked.mean()

        # "uncertainty" (also used for "gradnorm" as proxy)
        # L_i_weighted = 0.5 * exp(-2*log_sigma_i) * L_i + log_sigma_i
        precision = torch.exp(-2.0 * self.log_sigma)          # 1/sigma^2
        weighted = 0.5 * precision * stacked + self.log_sigma  # [n_tasks]
        return weighted.sum()

    def get_weights(self) -> torch.Tensor:
        """Return per-task weights (1/sigma = exp(-log_sigma)), shape [n_tasks]."""
        return torch.exp(-self.log_sigma)


# ---------------------------------------------------------------------------
# MTLTrainer
# ---------------------------------------------------------------------------

class MTLTrainer:
    """Trainer that combines PCGrad + dynamic task weighting."""

    def __init__(
        self,
        model: MTLModel,
        weighter: DynamicTaskWeighter,
        lr: float = 1e-4,
    ) -> None:
        self.model = model
        self.weighter = weighter
        self.gradient_surgery = GradientSurgery(model)
        self.optimizer = torch.optim.Adam(
            list(model.parameters()) + list(weighter.parameters()),
            lr=lr,
        )
        self._task_names: List[str] = list(model.task_heads.keys())

    def train_step(
        self,
        input_ids: torch.Tensor,
        targets_dict: Dict[str, torch.Tensor],
    ) -> Tuple[float, Dict[str, float]]:
        """
        One training step.
        Returns (total_loss_float, per_task_losses_dict).
        """
        self.optimizer.zero_grad()

        # 1. Compute per-task losses
        loss_dict = self.model.compute_all_losses(input_ids, targets_dict)
        per_task_losses = {k: v.item() for k, v in loss_dict.items()}

        # 2. Dynamic weighting to get scalar
        ordered_losses = [loss_dict[t] for t in self._task_names if t in loss_dict]
        total_loss = self.weighter(ordered_losses)

        # 3. Backprop
        total_loss.backward()

        # 4. Gradient surgery on backbone params
        backbone_params = [p for p in self.model.backbone.parameters() if p.requires_grad]

        if len(backbone_params) > 0:
            # Re-compute per-task gradients for surgery (retain_graph not needed; we already
            # have grads from the joint backward, but PCGrad needs per-task grads separately).
            # We do a fresh forward+grad for each task, then merge.
            self.optimizer.zero_grad()

            per_task_loss_dict: Dict[str, torch.Tensor] = {}
            for t in self._task_names:
                if t in targets_dict:
                    per_task_loss_dict[t] = self.model.compute_all_losses(
                        input_ids, {t: targets_dict[t]}
                    )[t]

            task_grads = self.gradient_surgery.compute_task_gradients(
                per_task_loss_dict, backbone_params
            )
            merged_grads = self.gradient_surgery.project_conflicting(task_grads)

            for p, g in zip(backbone_params, merged_grads):
                p.grad = g

            # Recompute weighter loss grads (weighter params don't need surgery)
            w_losses = [per_task_loss_dict[t] for t in self._task_names if t in per_task_loss_dict]
            w_total = self.weighter(w_losses)
            w_total.backward()

        self.optimizer.step()
        return total_loss.item(), per_task_losses

    def get_task_weights(self) -> Dict[str, float]:
        """Return per-task weights as dict."""
        weights = self.weighter.get_weights().detach()
        return {
            t: weights[i].item()
            for i, t in enumerate(self._task_names)
            if i < len(weights)
        }


# ---------------------------------------------------------------------------
# MTLConfig
# ---------------------------------------------------------------------------

@dataclass
class MTLConfig:
    d_model: int = 32
    vocab_size: int = 64
    n_layers: int = 2
    n_tasks: int = 3
    method: str = "uncertainty"
    lr: float = 1e-4
