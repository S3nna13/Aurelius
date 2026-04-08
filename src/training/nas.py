"""Neural Architecture Search via DARTS (Differentiable Architecture Search).

Implements DARTS-style differentiable architecture search where architecture
choices are represented as continuous weights optimized by gradient descent.

Key idea: Instead of discrete architecture choices, represent the architecture
as a weighted mixture of all candidate operations. Architecture weights (α) are
learned alongside model weights via bilevel optimization.

Reference: Liu et al. 2019 "DARTS: Differentiable Architecture Search"
           arXiv:1806.09055
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class MixedOperation(nn.Module):
    """A mixture of operations for differentiable architecture search.

    At each position, compute: output = sum_i (softmax(α)[i] * op_i(x))

    Args:
        operations: dict of {name: nn.Module}
    """

    def __init__(self, operations: dict[str, nn.Module]) -> None:
        super().__init__()
        self.ops = nn.ModuleDict(operations)
        # Architecture weights (unnormalized)
        self.arch_weights = nn.Parameter(torch.zeros(len(operations)))
        self._op_names = list(operations.keys())

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        """Compute weighted mixture of all operations.

        weights = softmax(arch_weights)
        output = sum_i weights[i] * ops[i](x)
        """
        weights = F.softmax(self.arch_weights, dim=0)
        out = None
        for i, name in enumerate(self._op_names):
            op_out = self.ops[name](x, *args, **kwargs)
            if out is None:
                out = weights[i] * op_out
            else:
                out = out + weights[i] * op_out
        return out

    def best_operation(self) -> str:
        """Return name of operation with highest arch weight."""
        with torch.no_grad():
            idx = int(self.arch_weights.argmax().item())
        return self._op_names[idx]

    def architecture_weights(self) -> dict[str, float]:
        """Return dict {op_name: weight} with softmax applied."""
        with torch.no_grad():
            weights = F.softmax(self.arch_weights, dim=0)
        return {name: float(weights[i].item()) for i, name in enumerate(self._op_names)}


class DARTSCell(nn.Module):
    """A DARTS-searchable transformer layer.

    Searches over:
    - Attention type: standard GQA vs sliding window (simplified here as two linear ops)
    - FFN type: SwiGLU vs simple linear

    For testability, operations are simplified to nn.Linear layers with different configs.
    In production, these would be full attention + FFN modules.
    """

    def __init__(self, d_model: int, n_candidates: int = 3) -> None:
        super().__init__()
        # n_candidates linear operations as proxy for different architectures
        ops = {
            f'op_{i}': nn.Linear(d_model, d_model, bias=(i % 2 == 0))
            for i in range(n_candidates)
        }
        self.mixed_op = MixedOperation(ops)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mixed_op(self.norm(x))

    def best_op(self) -> str:
        return self.mixed_op.best_operation()


class DARTSSearcher:
    """DARTS architecture search trainer.

    Maintains two optimizers:
    - model_optimizer: updates model weights (W) on training data
    - arch_optimizer: updates architecture weights (α) on validation data

    Bilevel optimization:
    - Inner loop: minimize train_loss wrt W (model weights)
    - Outer loop: minimize val_loss wrt α (architecture weights)
    """

    def __init__(
        self,
        model: nn.Module,
        arch_lr: float = 3e-4,
        model_lr: float = 1e-3,
        arch_weight_decay: float = 1e-3,
    ) -> None:
        # Separate model weights from arch weights
        arch_params = [p for n, p in model.named_parameters() if 'arch_weights' in n]
        model_params = [p for n, p in model.named_parameters() if 'arch_weights' not in n]

        self.arch_optimizer = torch.optim.Adam(
            arch_params, lr=arch_lr, weight_decay=arch_weight_decay
        )
        self.model_optimizer = torch.optim.Adam(model_params, lr=model_lr)

    def model_step(self, train_loss: torch.Tensor) -> float:
        """Update model weights on train loss. Returns loss float."""
        self.model_optimizer.zero_grad()
        train_loss.backward()
        self.model_optimizer.step()
        return float(train_loss.item())

    def arch_step(self, val_loss: torch.Tensor) -> float:
        """Update architecture weights on val loss. Returns loss float."""
        self.arch_optimizer.zero_grad()
        val_loss.backward()
        self.arch_optimizer.step()
        return float(val_loss.item())

    def get_best_architecture(self, model: nn.Module) -> dict[str, str]:
        """Return best operation for each MixedOperation in model.

        Walk model.named_modules(), find MixedOperation instances.
        Return {module_name: best_op_name}
        """
        result: dict[str, str] = {}
        for name, module in model.named_modules():
            if isinstance(module, MixedOperation):
                result[name] = module.best_operation()
        return result


class ArchitectureStats:
    """Track architecture weight evolution during search."""

    def __init__(self) -> None:
        self.history: list[dict] = []  # list of {step: n, weights: {name: float}}

    def record(self, step: int, mixed_op: MixedOperation) -> None:
        self.history.append({
            'step': step,
            'weights': mixed_op.architecture_weights(),
        })

    def dominant_op_over_time(self) -> list[str]:
        """Return list of best_operation at each recorded step."""
        result = []
        for entry in self.history:
            weights = entry['weights']
            best = max(weights, key=lambda k: weights[k])
            result.append(best)
        return result

    def convergence_step(self, threshold: float = 0.8) -> int | None:
        """Return first step where one operation has weight >= threshold.

        Returns None if never converged.
        """
        for entry in self.history:
            weights = entry['weights']
            if any(w >= threshold for w in weights.values()):
                return entry['step']
        return None
