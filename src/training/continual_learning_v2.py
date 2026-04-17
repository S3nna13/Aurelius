"""Advanced Continual Learning: Online EWC++ and A-GEM.

EWC++ (Schwarz et al. 2018) — online Fisher estimate via exponential moving average.
A-GEM (Chaudhry et al. 2019) — averaged gradient episodic memory projection.

References:
    https://arxiv.org/abs/1805.06370 (Progress & Compress / EWC++)
    https://arxiv.org/abs/1812.00420 (A-GEM)
"""
from __future__ import annotations

import random
from typing import Callable, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor


# ---------------------------------------------------------------------------
# Online Fisher Estimate (EWC++)
# ---------------------------------------------------------------------------

class OnlineFisherEstimate:
    """Maintains a running EMA of per-parameter squared gradients as Fisher estimate.

    Args:
        model: The neural network whose parameters are tracked.
        gamma: EMA decay factor. 0.9 means 10% update per step.
    """

    def __init__(self, model: nn.Module, gamma: float = 0.9) -> None:
        self.model = model
        self.gamma = gamma
        self.fisher: Dict[str, Tensor] = {
            name: torch.zeros_like(param.data)
            for name, param in model.named_parameters()
            if param.requires_grad
        }
        self.params_star: Dict[str, Tensor] = {
            name: param.data.clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

    def update(self, loss: Tensor) -> None:
        """Compute gradients of loss and update Fisher EMA + reference params.

        Args:
            loss: Scalar loss tensor to differentiate.
        """
        params = [p for _, p in self.model.named_parameters() if p.requires_grad]
        names = [n for n, p in self.model.named_parameters() if p.requires_grad]

        grads = torch.autograd.grad(
            loss,
            params,
            create_graph=False,
            allow_unused=True,
        )

        for name, grad, param in zip(names, grads, params):
            if grad is None:
                continue
            # EMA update: F_new = gamma * F_old + (1 - gamma) * grad^2
            self.fisher[name] = (
                self.gamma * self.fisher[name] + (1.0 - self.gamma) * grad.detach() ** 2
            )
            # Store current params as reference θ*
            self.params_star[name] = param.data.clone()

    def ewc_penalty(self, model: nn.Module) -> Tensor:
        """Compute EWC regularization penalty.

        Returns:
            Scalar tensor: 0.5 * sum_n fisher[n] * (param[n] - params_star[n])^2
        """
        penalty = torch.tensor(0.0)
        for name, param in model.named_parameters():
            if name not in self.fisher:
                continue
            diff = param - self.params_star[name]
            penalty = penalty + (self.fisher[name] * diff ** 2).sum()
        return 0.5 * penalty


# ---------------------------------------------------------------------------
# EWC++ Trainer
# ---------------------------------------------------------------------------

class EWCPlusPlusTrainer:
    """Trainer that applies Online EWC++ regularization.

    Args:
        model: The neural network to train.
        optimizer: Optimizer to use for parameter updates.
        ewc_lambda: Regularization strength for EWC penalty.
        gamma: EMA decay for Fisher estimate.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        ewc_lambda: float = 1000.0,
        gamma: float = 0.9,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.ewc_lambda = ewc_lambda
        self.fisher_estimate = OnlineFisherEstimate(model, gamma)

    def train_step(
        self, x: Tensor, loss_fn: Callable[[Tensor], Tensor]
    ) -> Dict[str, float]:
        """Execute one training step with EWC++ regularization.

        Args:
            x: Input tensor.
            loss_fn: Callable that receives model output and returns scalar loss.

        Returns:
            Dict with keys: 'task_loss', 'ewc_penalty', 'total_loss'.
        """
        self.model.train()
        self.optimizer.zero_grad()

        # Compute task loss
        output = self.model(x)
        task_loss: Tensor = loss_fn(output)

        # Compute EWC penalty
        penalty: Tensor = self.fisher_estimate.ewc_penalty(self.model)

        # Total loss
        total_loss: Tensor = task_loss + self.ewc_lambda * penalty

        # Backward and optimizer step
        total_loss.backward()
        self.optimizer.step()

        # Update Fisher estimate using task loss (recomputed without graph)
        self.optimizer.zero_grad()
        output2 = self.model(x)
        task_loss2: Tensor = loss_fn(output2)
        self.fisher_estimate.update(task_loss2)

        return {
            "task_loss": task_loss.item(),
            "ewc_penalty": penalty.item(),
            "total_loss": total_loss.item(),
        }


# ---------------------------------------------------------------------------
# Episodic Memory for A-GEM
# ---------------------------------------------------------------------------

class EpisodicMemory:
    """Stores past task examples for A-GEM gradient projection.

    Args:
        max_size: Maximum number of samples to store.
        d_input: Dimensionality of input tensors (used for type hints only).
    """

    def __init__(self, max_size: int = 100, d_input: int = 8) -> None:
        self.max_size = max_size
        self.d_input = d_input
        self.memory: List[Tensor] = []

    def add(self, x: Tensor) -> None:
        """Add a sample to memory, evicting the oldest if over capacity.

        Args:
            x: Input tensor to store (detached from computation graph).
        """
        self.memory.append(x.detach())
        if len(self.memory) > self.max_size:
            self.memory.pop(0)

    def sample(self, n: int) -> Optional[Tensor]:
        """Return n randomly sampled memory tensors stacked.

        Args:
            n: Number of samples to return.

        Returns:
            Stacked tensor of shape (k, d_input) where k = min(n, len(memory)),
            or None if memory is empty.
        """
        if not self.memory:
            return None
        k = min(n, len(self.memory))
        chosen = random.sample(self.memory, k)
        return torch.stack(chosen, dim=0)


# ---------------------------------------------------------------------------
# A-GEM Trainer
# ---------------------------------------------------------------------------

class AGEMTrainer:
    """Trainer that uses Averaged Gradient Episodic Memory (A-GEM) projection.

    Args:
        model: The neural network to train.
        optimizer: Optimizer to use for parameter updates.
        memory: EpisodicMemory instance holding past task samples.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        memory: EpisodicMemory,
    ) -> None:
        self.model = model
        self.optimizer = optimizer
        self.memory = memory

    def _project_gradient(
        self, current_grads: List[Tensor], ref_grads: List[Tensor]
    ) -> List[Tensor]:
        """Project current gradients to not increase reference (memory) loss.

        If dot(g_current, g_ref) >= 0, returns g_current unchanged.
        Otherwise projects: g_proj = g_c - (dot(g_c, g_r) / dot(g_r, g_r)) * g_r

        Args:
            current_grads: Gradients from current task.
            ref_grads: Gradients from episodic memory (reference).

        Returns:
            Projected gradient list with same structure as current_grads.
        """
        # Flatten all gradients into single vectors for dot product computation
        g_c = torch.cat([g.flatten() for g in current_grads])
        g_r = torch.cat([g.flatten() for g in ref_grads])

        dot_cg_rg = torch.dot(g_c, g_r)

        if dot_cg_rg.item() >= 0:
            return current_grads

        dot_rg_rg = torch.dot(g_r, g_r)
        # Avoid division by zero
        if dot_rg_rg.item() == 0:
            return current_grads

        scale = dot_cg_rg / dot_rg_rg

        projected: List[Tensor] = []
        for g_ci, g_ri in zip(current_grads, ref_grads):
            projected.append(g_ci - scale * g_ri)

        return projected

    def train_step(
        self,
        x: Tensor,
        y: Tensor,
        loss_fn: Callable[[Tensor, Tensor], Tensor],
        n_memory_samples: int = 10,
    ) -> Dict[str, float]:
        """Execute one A-GEM training step with optional gradient projection.

        Args:
            x: Input tensor for current task.
            y: Target tensor for current task.
            loss_fn: Loss function taking (predictions, targets) -> scalar.
            n_memory_samples: Number of memory samples for reference gradient.

        Returns:
            Dict with keys: 'task_loss', 'projected'.
        """
        self.model.train()

        # Compute current task gradients
        self.optimizer.zero_grad()
        output = self.model(x)
        task_loss: Tensor = loss_fn(output, y)
        task_loss.backward()

        # Collect current gradients
        current_grads: List[Tensor] = []
        params_with_grad: List[nn.Parameter] = []
        for param in self.model.parameters():
            if param.requires_grad and param.grad is not None:
                current_grads.append(param.grad.clone())
                params_with_grad.append(param)

        projected = False
        memory_batch = self.memory.sample(n_memory_samples)

        if memory_batch is not None:
            # Compute reference gradients from episodic memory
            self.optimizer.zero_grad()
            mem_output = self.model(memory_batch)
            # For reference loss, we compute unsupervised loss (mean output proxy)
            # using the memory targets derived from the memory outputs themselves
            mem_loss: Tensor = mem_output.float().mean()
            # Use proper reconstruction: just use a self-supervised proxy
            # A-GEM paper uses labels stored with memory; here we use output variance
            # as a proxy signal that creates non-trivial gradients
            mem_loss = (mem_output ** 2).mean()
            mem_loss.backward()

            ref_grads: List[Tensor] = []
            for param in params_with_grad:
                if param.grad is not None:
                    ref_grads.append(param.grad.clone())
                else:
                    ref_grads.append(torch.zeros_like(param))

            # Project current gradients
            projected_grads = self._project_gradient(current_grads, ref_grads)

            # Check if projection actually happened
            g_c_flat = torch.cat([g.flatten() for g in current_grads])
            g_proj_flat = torch.cat([g.flatten() for g in projected_grads])
            projected = not torch.allclose(g_c_flat, g_proj_flat)

            # Assign projected gradients back to parameters
            self.optimizer.zero_grad()
            for param, grad in zip(params_with_grad, projected_grads):
                param.grad = grad.clone()
        else:
            # No memory — just keep the current task gradients
            # Re-set gradients (they were cleared by last zero_grad? No — we need to restore)
            self.optimizer.zero_grad()
            for param, grad in zip(params_with_grad, current_grads):
                param.grad = grad.clone()

        self.optimizer.step()

        return {
            "task_loss": task_loss.item(),
            "projected": projected,
        }
