"""Continual Learning: EWC (Elastic Weight Consolidation) and PackNet.

Implements:
  - FisherInformationEstimator  : diagonal Fisher via squared gradients
  - EWCRegularizer              : anchor parameters + quadratic penalty
  - OnlineEWC                   : running-average Fisher across tasks
  - PackNetManager              : magnitude pruning + gradient masking
  - ProgressiveNNColumn         : independent columns + lateral adapters
  - ContinualTrainer            : unified training interface
  - ContinualConfig             : hyperparameter dataclass

References:
    Kirkpatrick et al. (2017) "Overcoming catastrophic forgetting"
    Mallya & Lazebnik (2018)  "PackNet"
    Rusu et al. (2016)        "Progressive Neural Networks"
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import Callable, Dict, Iterator, List, Optional, Tuple

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# FisherInformationEstimator
# ---------------------------------------------------------------------------

class FisherInformationEstimator:
    """Estimates the diagonal of the Fisher Information Matrix.

    F_i = E[(d log p / d theta_i)^2]

    Approximated by accumulating squared gradients of the loss (treated as
    -log p) over a finite number of examples.

    Args:
        model     : The nn.Module whose parameters will be tracked.
        n_samples : Maximum number of individual examples to process.
    """

    def __init__(self, model: nn.Module, n_samples: int = 200) -> None:
        self.model = model
        self.n_samples = n_samples

    def compute_diagonal_fisher(
        self,
        dataloader_fn: Callable,
        loss_fn: Callable,
    ) -> Dict[str, Tensor]:
        """Accumulate squared gradients to estimate diagonal Fisher.

        Args:
            dataloader_fn : Zero-argument callable that yields (inputs, targets)
                            batches when iterated.
            loss_fn       : (outputs, targets) -> scalar Tensor.

        Returns:
            dict mapping param name -> Fisher diagonal Tensor (same shape as param).
        """
        self.model.train(False)

        # Initialise accumulators to zero.
        fisher: Dict[str, Tensor] = {
            name: torch.zeros_like(param, requires_grad=False)
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }

        samples_seen = 0
        for inputs, targets in dataloader_fn():
            if samples_seen >= self.n_samples:
                break

            batch_size = inputs.shape[0] if hasattr(inputs, "shape") else 1
            remaining = self.n_samples - samples_seen
            if batch_size > remaining:
                inputs = inputs[:remaining]
                targets = targets[:remaining]
                batch_size = remaining

            self.model.zero_grad()
            outputs = self.model(inputs)
            loss = loss_fn(outputs, targets)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher[name] += (param.grad.detach() ** 2) * batch_size

            samples_seen += batch_size

        # Normalise by total samples seen.
        if samples_seen > 0:
            for name in fisher:
                fisher[name] /= samples_seen

        self.model.zero_grad()
        return fisher


# ---------------------------------------------------------------------------
# EWCRegularizer
# ---------------------------------------------------------------------------

class EWCRegularizer:
    """Elastic Weight Consolidation penalty term.

    After finishing a task, call ``consolidate`` to save the current parameters
    (theta*) and the corresponding Fisher diagonals.  During subsequent task
    training include ``ewc_loss`` in the optimisation objective.

    Args:
        model      : The nn.Module being trained.
        lambda_ewc : Strength of the EWC penalty.
    """

    def __init__(self, model: nn.Module, lambda_ewc: float = 1000.0) -> None:
        self.model = model
        self.lambda_ewc = lambda_ewc

        # List of (theta_star, fisher) tuples, one per consolidated task.
        self._anchors: List[Tuple[Dict[str, Tensor], Dict[str, Tensor]]] = []

    def consolidate(self, fisher: Dict[str, Tensor]) -> None:
        """Save the current model parameters and Fisher diagonals.

        Args:
            fisher : Diagonal Fisher estimates from FisherInformationEstimator.
        """
        theta_star: Dict[str, Tensor] = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }
        fisher_copy = {name: f.detach().clone() for name, f in fisher.items()}
        self._anchors.append((theta_star, fisher_copy))

    def penalty(self, model: nn.Module) -> Tensor:
        """Compute the EWC quadratic penalty across all consolidated tasks.

        penalty = sum_{tasks} sum_i  F_i * (theta_i - theta*_i)^2

        Args:
            model : nn.Module with the current parameters.

        Returns:
            Scalar tensor.
        """
        if not self._anchors:
            params = list(model.parameters())
            return torch.zeros(1, device=params[0].device if params else "cpu")

        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        param_dict = dict(model.named_parameters())

        for theta_star, fisher in self._anchors:
            for name, f in fisher.items():
                if name not in param_dict:
                    continue
                theta_curr = param_dict[name]
                theta_ref = theta_star[name].to(theta_curr.device)
                f_dev = f.to(theta_curr.device)
                loss = loss + (f_dev * (theta_curr - theta_ref) ** 2).sum()

        return loss

    def ewc_loss(self, task_loss: Tensor, model: nn.Module) -> Tensor:
        """Combined task loss + EWC regularisation.

        Returns:
            task_loss + lambda_ewc * penalty(model)
        """
        return task_loss + self.lambda_ewc * self.penalty(model)


# ---------------------------------------------------------------------------
# OnlineEWC
# ---------------------------------------------------------------------------

class OnlineEWC(EWCRegularizer):
    """EWC with a running (online) Fisher estimate.

    Instead of storing per-task Fisher matrices, maintain a single running
    estimate that blends old and new Fisher:

        F_running = gamma * F_old + (1 - gamma) * F_new

    Args:
        model      : The nn.Module being trained.
        lambda_ewc : EWC penalty strength.
        gamma      : Decay factor for old Fisher (1.0 = keep old fully).
    """

    def __init__(
        self,
        model: nn.Module,
        lambda_ewc: float = 1000.0,
        gamma: float = 1.0,
    ) -> None:
        super().__init__(model, lambda_ewc)
        self.gamma = gamma
        self._running_fisher: Optional[Dict[str, Tensor]] = None
        self._running_theta_star: Optional[Dict[str, Tensor]] = None

    # Override penalty to use the running anchor.
    def penalty(self, model: nn.Module) -> Tensor:
        if self._running_fisher is None:
            params = list(model.parameters())
            return torch.zeros(1, device=params[0].device if params else "cpu")

        loss = torch.tensor(0.0, device=next(model.parameters()).device)
        param_dict = dict(model.named_parameters())

        for name, f in self._running_fisher.items():
            if name not in param_dict or self._running_theta_star is None:
                continue
            theta_curr = param_dict[name]
            theta_ref = self._running_theta_star[name].to(theta_curr.device)
            f_dev = f.to(theta_curr.device)
            loss = loss + (f_dev * (theta_curr - theta_ref) ** 2).sum()

        return loss

    def consolidate_online(self, fisher: Dict[str, Tensor], task_id: int) -> None:
        """Update the running Fisher and save current theta*.

        F_running = gamma * F_old + (1 - gamma) * F_new

        Args:
            fisher  : New diagonal Fisher for the just-completed task.
            task_id : Integer task identifier (informational only).
        """
        if self._running_fisher is None:
            self._running_fisher = {
                name: f.detach().clone() for name, f in fisher.items()
            }
        else:
            for name, f_new in fisher.items():
                if name in self._running_fisher:
                    self._running_fisher[name] = (
                        self.gamma * self._running_fisher[name]
                        + (1.0 - self.gamma) * f_new.detach()
                    )
                else:
                    self._running_fisher[name] = f_new.detach().clone()

        # Save current parameters as the new anchor.
        self._running_theta_star = {
            name: param.detach().clone()
            for name, param in self.model.named_parameters()
            if param.requires_grad
        }


# ---------------------------------------------------------------------------
# PackNetManager
# ---------------------------------------------------------------------------

class PackNetManager:
    """Progressive network pruning and weight freezing (PackNet).

    Workflow per task:
        1. Train task t on ALL unfrozen weights.
        2. Call prune_for_task(t) — keep top (1-prune_fraction) free weights
           by magnitude; zero the rest.
        3. Call freeze_task_weights(t) — grad hooks zero gradients on any
           weight allocated to a completed task.
        4. Move to task t+1.

    Args:
        model          : The nn.Module to manage.
        prune_fraction : Fraction of currently free weights to prune away.
    """

    def __init__(self, model: nn.Module, prune_fraction: float = 0.5) -> None:
        self.model = model
        self.prune_fraction = prune_fraction

        # task_masks[task_id][param_name] -> binary bool Tensor
        self.task_masks: Dict[int, Dict[str, Tensor]] = {}

        # Accumulated mask of all weights allocated to any prior task.
        self._allocated_mask: Dict[str, Tensor] = {
            name: torch.zeros_like(param, dtype=torch.bool)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        # Gradient hook handles.
        self._hook_handles: List = []

    def prune_for_task(self, task_id: int) -> None:
        """Assign highest-magnitude free weights to task_id; zero the rest.

        Args:
            task_id : Integer identifier for the task being completed.
        """
        task_mask: Dict[str, Tensor] = {}

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue

            allocated = self._allocated_mask.get(
                name, torch.zeros_like(param, dtype=torch.bool)
            )
            free_mask = ~allocated
            free_count = int(free_mask.sum().item())

            if free_count == 0:
                task_mask[name] = torch.zeros_like(param, dtype=torch.bool)
                continue

            n_keep = max(1, int(math.ceil(free_count * (1.0 - self.prune_fraction))))
            n_keep = min(n_keep, free_count)

            # Select top-n_keep free weights by absolute value.
            free_vals = param.detach().abs().flatten()
            # Indices within the full flat param of free positions.
            free_indices = free_mask.flatten().nonzero(as_tuple=False).squeeze(1)
            free_abs = free_vals[free_indices]

            # Sort free positions by magnitude descending; take top n_keep.
            _, order = free_abs.sort(descending=True)
            keep_local = order[:n_keep]  # indices within free_indices
            keep_global = free_indices[keep_local]  # indices in flat param

            keep_flat = torch.zeros(param.numel(), dtype=torch.bool, device=param.device)
            keep_flat[keep_global] = True
            keep_mask = keep_flat.view_as(param)

            # Zero out pruned free weights.
            pruned_mask = free_mask & ~keep_mask
            with torch.no_grad():
                param[pruned_mask] = 0.0

            task_mask[name] = keep_mask
            self._allocated_mask[name] = allocated | keep_mask

        self.task_masks[task_id] = task_mask

    def freeze_task_weights(self, task_id: int) -> None:
        """Register gradient hooks that zero gradients on allocated weights.

        Args:
            task_id : Task whose weights (plus all prior) to protect.
        """
        self.unfreeze_all()

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            frozen = self._allocated_mask.get(name)
            if frozen is None or not frozen.any():
                continue

            def _make_hook(mask: Tensor):
                def _hook(grad: Tensor) -> Tensor:
                    return grad * (~mask).to(grad.dtype)
                return _hook

            handle = param.register_hook(_make_hook(frozen.to(param.device)))
            self._hook_handles.append(handle)

    def unfreeze_all(self) -> None:
        """Remove all gradient hooks registered by freeze_task_weights."""
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def apply_masks(self, model: nn.Module, task_id: int) -> None:
        """Zero all weights not belonging to task_id (for inference isolation).

        Args:
            model   : nn.Module to apply masks to.
            task_id : Task whose weights to preserve.
        """
        if task_id not in self.task_masks:
            warnings.warn(f"PackNetManager: no mask for task_id={task_id}.")
            return

        mask = self.task_masks[task_id]
        with torch.no_grad():
            for name, param in model.named_parameters():
                if name in mask:
                    param.mul_(mask[name].float().to(param.device))


# ---------------------------------------------------------------------------
# ProgressiveNNColumn
# ---------------------------------------------------------------------------

class _LateralAdapter(nn.Module):
    """Small linear adapter connecting one column's output to another's input."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.proj = nn.Linear(d_model, d_model, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=1e-3)

    def forward(self, x: Tensor) -> Tensor:
        return self.proj(x)


class _Column(nn.Module):
    """Two-layer feed-forward column for one task."""

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Linear(d_model * 2, d_model),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class ProgressiveNNColumn(nn.Module):
    """Progressive Neural Network with lateral connections.

    Each task gets its own column (sub-network).  Task t's output:

        h_t(x) = column_t(x) + sum_{k < t} lateral_{k->t}(h_k(x))

    where h_k(x) is column k evaluated on x.

    Args:
        d_model : Feature dimension.
        n_tasks : Number of tasks to pre-allocate columns for.
    """

    def __init__(self, d_model: int, n_tasks: int) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_tasks = n_tasks

        self.columns = nn.ModuleList([_Column(d_model) for _ in range(n_tasks)])

        # lateral_connections[t] is a ModuleList with t adapters (from col 0..t-1).
        self.lateral_connections = nn.ModuleList(
            [
                nn.ModuleList([_LateralAdapter(d_model) for _ in range(t)])
                for t in range(n_tasks)
            ]
        )

    def forward(self, x: Tensor, task_id: int) -> Tensor:
        """Forward pass for task_id.

        Args:
            x       : [B, T, d_model] input tensor.
            task_id : Which column to use as primary.

        Returns:
            [B, T, d_model] output tensor.
        """
        if task_id >= self.n_tasks:
            raise ValueError(f"task_id={task_id} >= n_tasks={self.n_tasks}")

        # Compute prior column outputs without gradient accumulation.
        prior_outputs: List[Tensor] = []
        for k in range(task_id):
            with torch.no_grad():
                h_k = self.columns[k](x)
            prior_outputs.append(h_k)

        # Primary column.
        h_t = self.columns[task_id](x)

        # Add lateral contributions.
        laterals = self.lateral_connections[task_id]
        for k, h_k in enumerate(prior_outputs):
            h_t = h_t + laterals[k](h_k)

        return h_t


# ---------------------------------------------------------------------------
# ContinualTrainer
# ---------------------------------------------------------------------------

class ContinualTrainer:
    """Unified training loop for continual-learning strategies.

    Supported strategies:
        "ewc"     -- EWCRegularizer applied after first task.
        "packnet" -- PackNetManager with prune + freeze after each task.
        "naive"   -- Standard SGD with no forgetting mitigation.

    Args:
        model    : nn.Module to train.
        strategy : One of "ewc", "packnet", "naive".
        lr       : Learning rate for SGD optimiser.
    """

    def __init__(
        self,
        model: nn.Module,
        strategy: str = "ewc",
        lr: float = 1e-3,
    ) -> None:
        if strategy not in ("ewc", "packnet", "naive"):
            raise ValueError(f"Unknown strategy: {strategy!r}")

        self.model = model
        self.strategy = strategy
        self.lr = lr

        self.ewc: Optional[EWCRegularizer] = (
            EWCRegularizer(model) if strategy == "ewc" else None
        )
        self.packnet: Optional[PackNetManager] = (
            PackNetManager(model) if strategy == "packnet" else None
        )
        self._completed_tasks: List[int] = []

    def train_task(
        self,
        task_id: int,
        data_fn: Callable,
        n_steps: int,
    ) -> List[float]:
        """Train on one task for n_steps gradient steps.

        Args:
            task_id : Integer task identifier.
            data_fn : Zero-argument callable returning an iterable of
                      (inputs, targets) pairs.
            n_steps : Number of gradient steps.

        Returns:
            List of per-step task loss values (floats).
        """
        self.model.train()
        optimiser = torch.optim.SGD(self.model.parameters(), lr=self.lr)
        criterion = nn.MSELoss()
        loss_history: List[float] = []

        def _fresh_iter() -> Iterator:
            return iter(data_fn())

        data_iter = _fresh_iter()

        for _ in range(n_steps):
            try:
                inputs, targets = next(data_iter)
            except StopIteration:
                data_iter = _fresh_iter()
                inputs, targets = next(data_iter)

            optimiser.zero_grad()
            outputs = self.model(inputs)
            task_loss = criterion(outputs, targets)

            if self.strategy == "ewc" and self.ewc is not None:
                total_loss = self.ewc.ewc_loss(task_loss, self.model)
            else:
                total_loss = task_loss

            total_loss.backward()
            optimiser.step()
            loss_history.append(task_loss.item())

        # Post-task consolidation.
        if self.strategy == "ewc" and self.ewc is not None:
            estimator = FisherInformationEstimator(self.model, n_samples=50)

            def _wrapped() -> Iterator:
                it = _fresh_iter()
                for _ in range(10):
                    try:
                        yield next(it)
                    except StopIteration:
                        return

            fisher = estimator.compute_diagonal_fisher(
                dataloader_fn=_wrapped,
                loss_fn=criterion,
            )
            self.ewc.consolidate(fisher)

        elif self.strategy == "packnet" and self.packnet is not None:
            self.packnet.prune_for_task(task_id)
            self.packnet.freeze_task_weights(task_id)

        self._completed_tasks.append(task_id)
        return loss_history

    def evaluate_forgetting(
        self,
        task_id: int,
        metric_fn: Callable,
    ) -> float:
        """Compute a performance metric on a given task.

        Args:
            task_id   : Task to assess (informational; passed to metric_fn).
            metric_fn : (model) -> float.  Higher = better.

        Returns:
            Float metric value.
        """
        self.model.train(False)
        with torch.no_grad():
            metric = metric_fn(self.model)
        return float(metric)


# ---------------------------------------------------------------------------
# ContinualConfig
# ---------------------------------------------------------------------------

@dataclass
class ContinualConfig:
    """Hyperparameters for continual learning experiments."""

    lambda_ewc: float = 1000.0
    """EWC regularisation strength."""

    gamma: float = 0.9
    """Fisher decay factor for OnlineEWC."""

    prune_fraction: float = 0.5
    """Fraction of free weights to prune per task in PackNet."""

    n_tasks: int = 3
    """Total number of tasks in the sequence."""

    strategy: str = "ewc"
    """Training strategy: 'ewc' | 'packnet' | 'naive'."""
