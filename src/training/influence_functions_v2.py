"""Influence Functions v2: LiSSA and DataInf approximations.

Provides scalable influence estimation for neural networks:
- LiSSA: Pearlmutter HVP with power-series Hessian inverse approximation
- DataInf: Diagonal Hessian approximation (Kwon et al. 2023)
- GradientSimilarity: Fast proxy via gradient dot products

References:
    Koh & Liang 2017 — https://arxiv.org/abs/1703.04730 (Influence Functions)
    Agarwal et al. 2017 — https://arxiv.org/abs/1602.03943 (LiSSA)
    Kwon et al. 2023 — https://arxiv.org/abs/2307.12580 (DataInf)
"""

from __future__ import annotations

import random
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# HessianVectorProduct
# ---------------------------------------------------------------------------


class HessianVectorProduct:
    """Computes exact Hessian-vector products via double backprop.

    Args:
        model: The neural network model.
        loss_fn: Callable that takes ``(model, batch)`` and returns a scalar loss.
    """

    def __init__(self, model: nn.Module, loss_fn: Callable) -> None:
        self.model = model
        self.loss_fn = loss_fn

    def hvp(self, batch: Tensor, vector: dict[str, Tensor]) -> dict[str, Tensor]:
        """Compute H * v where v is a vector in parameter space.

        Args:
            batch: A single batch passed directly to ``loss_fn(model, batch)``.
            vector: Dict mapping parameter names to tensors of the same shape,
                representing the vector to multiply with the Hessian.

        Returns:
            Dict ``{name: second_grad}`` for each named parameter.
        """
        # Step 1: forward pass with graph creation enabled
        loss = self.loss_fn(self.model, batch)

        # Step 2: compute first-order gradients (retain graph for second pass)
        param_list = [(name, p) for name, p in self.model.named_parameters() if p.requires_grad]
        params = [p for _, p in param_list]

        first_grads = torch.autograd.grad(
            loss,
            params,
            create_graph=True,
            allow_unused=True,
        )

        # Replace None gradients with zeros
        first_grads = [
            g if g is not None else torch.zeros_like(p) for g, p in zip(first_grads, params)
        ]

        # Step 3: dot product of first_grads with vector
        grad_dot_v = sum(
            (first_grads[i] * vector.get(name, torch.zeros_like(params[i]))).sum()
            for i, (name, _) in enumerate(param_list)
        )

        # Step 4: second-order gradient of the dot product
        second_grads = torch.autograd.grad(
            grad_dot_v,
            params,
            allow_unused=True,
        )

        result: dict[str, Tensor] = {}
        for (name, _), sg in zip(param_list, second_grads):
            if sg is None:
                result[name] = torch.zeros_like(vector.get(name, torch.zeros(1)))
            else:
                result[name] = sg.detach()

        return result


# ---------------------------------------------------------------------------
# LiSSAInfluence
# ---------------------------------------------------------------------------


class LiSSAInfluence:
    """Approximate H^{-1} * v using the LiSSA recursive power series.

    LiSSA approximates the inverse Hessian-vector product by the recursion:
        h_0 = v / scale
        h_{t+1} = v + h_t - HVP(batch_t) * h_t / scale

    Args:
        model: The neural network model.
        damping: Regularisation added as ``damping * I`` to stabilise H + λI.
        scale: LiSSA scale factor; divides the curvature contribution each step.
        n_iterations: Number of power series terms (recursion depth).
        n_samples: Number of independent LiSSA estimates to average.
    """

    def __init__(
        self,
        model: nn.Module,
        damping: float = 0.01,
        scale: float = 10.0,
        n_iterations: int = 10,
        n_samples: int = 1,
    ) -> None:
        self.model = model
        self.damping = damping
        self.scale = scale
        self.n_iterations = n_iterations
        self.n_samples = n_samples

    def estimate(
        self,
        train_batches: list[Tensor],
        loss_fn: Callable,
        test_grad: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Estimate H^{-1} * test_grad via LiSSA.

        Args:
            train_batches: List of batches used for stochastic HVP computation.
            loss_fn: Callable ``(model, batch) -> scalar``.
            test_grad: Dict mapping parameter names to gradient tensors.

        Returns:
            Estimated ``H^{-1} * test_grad`` as a parameter dict.
        """
        hvp_computer = HessianVectorProduct(self.model, loss_fn)

        # Collect parameter names in a stable order
        param_names = [name for name, p in self.model.named_parameters() if p.requires_grad]

        # Accumulator for averaging across samples
        accum: dict[str, Tensor] = {
            name: torch.zeros_like(test_grad[name]) for name in param_names if name in test_grad
        }
        # For params missing from test_grad, use zeros
        for name, p in self.model.named_parameters():
            if p.requires_grad and name not in accum:
                accum[name] = torch.zeros_like(p)

        if not train_batches:
            # No training data — return the test gradient unchanged
            return {
                name: test_grad.get(name, torch.zeros_like(accum[name])).clone()
                for name in param_names
            }

        for _ in range(self.n_samples):
            # Initialise estimate as v (test_grad)
            h_est: dict[str, Tensor] = {
                name: test_grad.get(name, torch.zeros_like(accum[name])).clone().float()
                for name in param_names
            }

            # Shuffle batches for this sample
            indices = list(range(len(train_batches)))
            random.shuffle(indices)

            for i in range(self.n_iterations):
                batch = train_batches[indices[i % len(indices)]]

                # HVP: Hv (with damping via the recursion update)
                hvp_result = hvp_computer.hvp(batch, h_est)

                # Recursion: h = v + (1 - damping/scale) * h - hvp / scale
                new_h: dict[str, Tensor] = {}
                for name in param_names:
                    v_n = test_grad.get(name, torch.zeros_like(accum[name])).float()
                    h_n = h_est[name]
                    hvp_n = hvp_result.get(name, torch.zeros_like(h_n)).float()
                    # Standard LiSSA update with damping absorbed:
                    # h = v + h * (1 - damping) - hvp / scale
                    new_h[name] = v_n + h_n * (1.0 - self.damping) - hvp_n / self.scale
                h_est = new_h

            # Accumulate
            for name in param_names:
                accum[name] = accum[name] + h_est[name].detach()

        # Average over samples
        result: dict[str, Tensor] = {name: accum[name] / self.n_samples for name in param_names}
        return result


# ---------------------------------------------------------------------------
# DataInfInfluence
# ---------------------------------------------------------------------------


class DataInfInfluence:
    """Diagonal Hessian approximation: H ≈ diag(mean(grad^2)).

    Fast but approximate. Based on Kwon et al. 2023 (DataInf).

    Args:
        model: The neural network model.
        damping: Added to the diagonal to prevent division by zero.
    """

    def __init__(self, model: nn.Module, damping: float = 0.01) -> None:
        self.model = model
        self.damping = damping

    def compute_diagonal_hessian(
        self,
        train_batches: list[Tensor],
        loss_fn: Callable,
    ) -> dict[str, Tensor]:
        """Estimate diagonal of Hessian as mean of squared gradients.

        Args:
            train_batches: List of batches to average over.
            loss_fn: Callable ``(model, batch) -> scalar``.

        Returns:
            Dict ``{param_name: diag_estimate}`` with same shapes as params.
        """
        param_names = [name for name, p in self.model.named_parameters() if p.requires_grad]
        params = [p for p in self.model.parameters() if p.requires_grad]

        accum: dict[str, Tensor] = {
            name: torch.zeros_like(p) for name, p in zip(param_names, params)
        }

        if not train_batches:
            return accum

        for batch in train_batches:
            # Zero existing gradients
            self.model.zero_grad()
            loss = loss_fn(self.model, batch)
            loss.backward()

            for name, p in zip(param_names, params):
                if p.grad is not None:
                    accum[name] = accum[name] + p.grad.detach() ** 2

        # Average over batches
        n = len(train_batches)
        diag_H: dict[str, Tensor] = {name: accum[name] / n for name in param_names}

        # Clear gradients
        self.model.zero_grad()
        return diag_H

    def estimate(
        self,
        train_batches: list[Tensor],
        loss_fn: Callable,
        test_grad: dict[str, Tensor],
    ) -> dict[str, Tensor]:
        """Estimate H^{-1} * test_grad using diagonal approximation.

        Args:
            train_batches: List of batches used to compute the diagonal Hessian.
            loss_fn: Callable ``(model, batch) -> scalar``.
            test_grad: Dict mapping parameter names to gradient tensors.

        Returns:
            Dict ``{name: test_grad[name] / (diag_H[name] + damping)}``.
        """
        diag_H = self.compute_diagonal_hessian(train_batches, loss_fn)

        result: dict[str, Tensor] = {}
        for name, diag in diag_H.items():
            tg = test_grad.get(name, torch.zeros_like(diag))
            result[name] = tg / (diag + self.damping)

        return result


# ---------------------------------------------------------------------------
# GradientSimilarity
# ---------------------------------------------------------------------------


class GradientSimilarity:
    """Fast proxy for influence via gradient dot products.

    Influence ≈ dot product of training and test gradients.

    Args:
        model: The neural network model.
    """

    def __init__(self, model: nn.Module) -> None:
        self.model = model

    def compute_train_grads(
        self,
        train_batches: list[Tensor],
        loss_fn: Callable,
    ) -> list[dict[str, Tensor]]:
        """Compute per-batch gradients.

        Args:
            train_batches: List of batches.
            loss_fn: Callable ``(model, batch) -> scalar``.

        Returns:
            List of gradient dicts, one per batch.
        """
        param_names = [name for name, p in self.model.named_parameters() if p.requires_grad]
        params = [p for p in self.model.parameters() if p.requires_grad]

        all_grads: list[dict[str, Tensor]] = []

        for batch in train_batches:
            self.model.zero_grad()
            loss = loss_fn(self.model, batch)
            loss.backward()

            grad_dict: dict[str, Tensor] = {}
            for name, p in zip(param_names, params):
                if p.grad is not None:
                    grad_dict[name] = p.grad.detach().clone()
                else:
                    grad_dict[name] = torch.zeros_like(p)

            all_grads.append(grad_dict)

        self.model.zero_grad()
        return all_grads

    def influence_scores(
        self,
        train_grads: list[dict[str, Tensor]],
        test_grad: dict[str, Tensor],
    ) -> list[float]:
        """Compute gradient dot-product influence scores.

        Args:
            train_grads: List of gradient dicts from ``compute_train_grads``.
            test_grad: Gradient dict for the test example.

        Returns:
            List of scalar dot-product scores, one per training batch.
        """
        scores: list[float] = []

        # Flatten test_grad into a single vector
        test_flat = torch.cat([v.reshape(-1) for v in test_grad.values()])

        for grad_dict in train_grads:
            train_flat = torch.cat(
                [
                    grad_dict.get(name, torch.zeros_like(v)).reshape(-1)
                    for name, v in test_grad.items()
                ]
            )
            score = torch.dot(train_flat, test_flat).item()
            scores.append(score)

        return scores

    def top_k_influential(self, scores: list[float], k: int) -> list[int]:
        """Return indices of the top-k most influential examples by absolute value.

        Args:
            scores: List of scalar influence scores.
            k: Number of top examples to return.

        Returns:
            List of at most ``k`` indices, sorted by descending absolute score.
        """
        k = min(k, len(scores))
        indexed = sorted(enumerate(scores), key=lambda x: abs(x[1]), reverse=True)
        return [idx for idx, _ in indexed[:k]]
