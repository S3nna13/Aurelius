"""Learned optimizer using a small LSTM to predict parameter updates.

Implements the L2L (Learning to Learn) paradigm where instead of hand-designed
update rules (like Adam's momentum + adaptive LR), a small LSTM network is
trained to predict what gradient update to apply.

Reference: Andrychowicz et al. 2016, "Learning to learn by gradient descent by gradient descent"
"""
from __future__ import annotations

import math
from collections.abc import Callable

import torch
import torch.nn as nn


class GradientPreprocessor:
    """Preprocess gradients for the optimizer LSTM.

    Raw gradients have wildly varying magnitudes. Preprocess to:
    1. Log-scale the magnitude: (log(|g|+ε)/log(10)) / p
    2. Sign: sign(g)
    This creates a 2D feature per parameter that's scale-invariant.
    """

    @staticmethod
    def preprocess(grad: torch.Tensor, p: float = 10.0) -> torch.Tensor:
        """Preprocess a gradient tensor into scale-invariant features.

        Args:
            grad: any shape gradient tensor
            p: log-scale divisor (default 10.0)

        Returns:
            (numel, 2) tensor:
                col 0: log-scaled magnitude in [-1, 1]
                col 1: sign in {-1, 0, 1}
        """
        flat = grad.flatten()
        log_mag = torch.log(flat.abs() + 1e-8) / (p * math.log(10))
        log_mag = log_mag.clamp(-1, 1)
        sign = flat.sign()
        return torch.stack([log_mag, sign], dim=1)  # (numel, 2)


class LSTMOptimizer(nn.Module):
    """Small LSTM that predicts parameter updates from gradients.

    For each parameter, processes its gradient features through an LSTM
    and predicts the update delta.

    Args:
        input_size: gradient feature size (2 from preprocessor)
        hidden_size: LSTM hidden size (default 20)
        n_layers: LSTM layers (default 2)
    """

    def __init__(
        self,
        input_size: int = 2,
        hidden_size: int = 20,
        n_layers: int = 2,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True)
        # Output: predicted update (scale in log space, direction)
        self.output_proj = nn.Linear(hidden_size, 2)

    def forward(
        self,
        grad_features: torch.Tensor,  # (numel, seq=1, 2)
        state: tuple | None = None,
    ) -> tuple[torch.Tensor, tuple]:
        """Predict parameter updates from gradient features.

        Args:
            grad_features: (numel, 1, 2) — 1 gradient step as sequence
            state: LSTM hidden state (None for initial step)

        Returns:
            (updates, new_state):
            - updates: (numel,) parameter update deltas
            - new_state: new LSTM state
        """
        output, new_state = self.lstm(grad_features, state)
        # output: (numel, 1, hidden_size)
        proj = self.output_proj(output.squeeze(1))  # (numel, 2)
        log_scale = proj[:, 0]
        direction = proj[:, 1]
        updates = torch.exp(log_scale) * direction  # (numel,)
        return updates, new_state


class LearnedOptimizerWrapper:
    """Wraps a target model and applies learned optimizer updates.

    Maintains separate LSTM state per parameter group.
    The meta-optimizer is trained separately (meta-training loop).

    Args:
        lstm_optimizer: trained LSTMOptimizer to use for updates
        target_model: the model whose parameters will be updated
        max_grad_norm: gradient clipping norm (default 1.0)
    """

    def __init__(
        self,
        lstm_optimizer: LSTMOptimizer,
        target_model: nn.Module,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.lstm_opt = lstm_optimizer
        self.model = target_model
        self.max_grad_norm = max_grad_norm
        # Per-parameter LSTM states
        self._states: dict[str, tuple] = {}

    def step(self, loss: torch.Tensor) -> dict:
        """Compute gradients and apply learned optimizer update.

        1. loss.backward()
        2. For each parameter: preprocess gradient, run LSTM, get update
        3. Apply update: param.data -= update
        4. Zero gradients

        Args:
            loss: scalar loss tensor to differentiate

        Returns:
            dict with keys:
                'n_params_updated': number of parameters updated
                'mean_update_norm': mean norm of update vectors
        """
        loss.backward()

        # Clip gradients
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

        preprocessor = GradientPreprocessor()
        n_params_updated = 0
        update_norms = []

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.grad is None:
                    continue

                # Preprocess gradient: (numel, 2)
                features = preprocessor.preprocess(param.grad)
                # Add sequence dimension: (numel, 1, 2)
                features = features.unsqueeze(1)

                # Get LSTM state for this parameter
                state = self._states.get(name, None)

                # Run LSTM optimizer
                updates, new_state = self.lstm_opt(features, state)

                # Store updated state
                self._states[name] = new_state

                # Apply update
                param.data -= updates.view_as(param.data)

                n_params_updated += 1
                update_norms.append(updates.norm().item())

        # Zero gradients
        for param in self.model.parameters():
            param.grad = None

        mean_update_norm = float(sum(update_norms) / len(update_norms)) if update_norms else 0.0

        return {
            "n_params_updated": n_params_updated,
            "mean_update_norm": mean_update_norm,
        }

    def reset_states(self) -> None:
        """Reset LSTM states (call at start of each meta-training episode)."""
        self._states = {}


class MetaTrainingLoop:
    """Outer loop for meta-training the learned optimizer.

    The key insight: train the LSTM optimizer by unrolling it for K steps
    on a task and backpropagating through the entire unrolled optimization.

    For simplicity here: train the LSTM to minimize loss on simple quadratic tasks.

    Args:
        lstm_optimizer: the LSTMOptimizer to meta-train
        meta_lr: learning rate for the meta-optimizer (default 1e-3)
    """

    def __init__(
        self,
        lstm_optimizer: LSTMOptimizer,
        meta_lr: float = 1e-3,
    ) -> None:
        self.lstm_opt = lstm_optimizer
        self.meta_optimizer = torch.optim.Adam(lstm_optimizer.parameters(), lr=meta_lr)

    class _TaskLossModule(nn.Module):
        """Wrap a task loss closure so functional_call can swap task_model params."""

        def __init__(self, task_model: nn.Module, task_loss_fn: Callable[[], torch.Tensor]) -> None:
            super().__init__()
            self.task_model = task_model
            self.task_loss_fn = task_loss_fn

        def forward(self) -> torch.Tensor:
            return self.task_loss_fn()

    def meta_step(
        self,
        task_model: nn.Module,
        task_loss_fn: Callable[[], torch.Tensor],
        n_unroll: int = 5,
    ) -> float:
        """Unroll optimizer for n_unroll steps, accumulate loss, meta-update.

        Uses torch.func.functional_call to evaluate the task loss with a
        differentiable parameter dict. This preserves the computational graph
        from LSTM outputs -> updated params -> task loss -> meta_loss, so
        gradients flow back into the LSTM parameters.

        Args:
            task_model: the model to optimize in the inner loop
            task_loss_fn: callable() -> scalar loss (uses task_model internally)
            n_unroll: number of inner optimization steps to unroll

        Returns:
            meta_loss as a float
        """
        preprocessor = GradientPreprocessor()
        states: dict[str, tuple] = {}
        loss_module = self._TaskLossModule(task_model, task_loss_fn)
        meta_loss: torch.Tensor | None = None

        # Maintain a differentiable parameter dict (updated each step via LSTM)
        params: dict[str, torch.Tensor] = {
            f"task_model.{name}": param.detach().clone().requires_grad_(True)
            for name, param in task_model.named_parameters()
        }
        params.update(
            {
                f"task_model.{name}": buffer
                for name, buffer in task_model.named_buffers()
            }
        )

        for _ in range(n_unroll):
            loss = torch.func.functional_call(loss_module, params, ())
            meta_loss = loss if meta_loss is None else meta_loss + loss

            # Compute gradients of loss w.r.t. current differentiable params
            param_items = list(task_model.named_parameters())
            param_names = [f"task_model.{name}" for name, _ in param_items]
            param_list = [params[name] for name in param_names]
            grads = torch.autograd.grad(
                loss,
                param_list,
                create_graph=True,
                allow_unused=True,
            )

            # Apply learned optimizer: LSTM maps grad features -> updates
            new_params = {
                f"task_model.{name}": buffer
                for name, buffer in task_model.named_buffers()
            }
            for name, grad in zip(param_names, grads):
                param = params[name]
                if grad is None:
                    new_params[name] = param
                    continue

                # Preprocess gradient (detach to avoid double-differentiating preprocessor)
                features = preprocessor.preprocess(grad.detach()).unsqueeze(1)

                state = states.get(name, None)
                updates, new_state = self.lstm_opt(features, state)
                states[name] = new_state

                # Differentiable update: gradient flows from meta_loss -> updates -> LSTM
                new_params[name] = param - updates.view_as(param)

            params = new_params

        # Meta-update
        self.meta_optimizer.zero_grad()
        if meta_loss is None:
            return 0.0
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()
