"""Influence Functions for data attribution (Koh & Liang 2017 + LiSSA approximation).

Estimates the influence of each training example on a test loss via:
    Influence(z_i, z_test) ≈ -grad_test^T H^{-1} grad_train_i

where H is the Hessian of the training loss w.r.t. model parameters and
H^{-1} is approximated using the LiSSA (Linear time Stochastic Second-Order
Algorithm) recursive estimator.

Usage::

    config = InfluenceConfig(n_recursion_depth=10, top_k=5)
    scores = compute_influence_scores(
        model, loss_fn, train_dataset, test_inputs, test_targets, config
    )
    indices, values = top_influential_examples(scores, k=config.top_k, mode="harmful")
"""

from __future__ import annotations

import random
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class InfluenceConfig:
    """Hyperparameters for influence function computation."""

    n_recursion_depth: int = 10  # LiSSA recursion depth
    recursion_scale: float = 0.1  # LiSSA damping / scale factor
    damping: float = 0.01  # Hessian damping λI added to H
    n_samples: int = 16  # samples used for stochastic HVP
    top_k: int = 5  # top-k influential examples to return


# ---------------------------------------------------------------------------
# Gradient computation
# ---------------------------------------------------------------------------


def compute_grad(
    model: nn.Module,
    loss_fn: Callable,
    inputs: Tensor,
    targets: Tensor,
    params: list[nn.Parameter] | None = None,
) -> list[Tensor]:
    """Compute gradient of loss w.r.t. params.

    Args:
        model: The neural network.
        loss_fn: Callable(model, inputs, targets) -> scalar loss tensor.
        inputs: Input tensor for the forward pass.
        targets: Target tensor for loss computation.
        params: List of parameters to differentiate w.r.t.
                Defaults to all model parameters that require grad.

    Returns:
        List of gradient tensors matching the structure of ``params``.
        Gradients are detached and zero-filled where ``None`` would appear.
    """
    if params is None:
        params = [p for p in model.parameters() if p.requires_grad]

    # Zero existing gradients to avoid accumulation
    for p in params:
        if p.grad is not None:
            p.grad.detach_()
            p.grad.zero_()

    loss = loss_fn(model, inputs, targets)
    grads = torch.autograd.grad(
        loss,
        params,
        create_graph=False,
        allow_unused=True,
    )

    result: list[Tensor] = []
    for g, p in zip(grads, params):
        if g is None:
            result.append(torch.zeros_like(p))
        else:
            result.append(g.detach())
    return result


# ---------------------------------------------------------------------------
# Hessian-vector product
# ---------------------------------------------------------------------------


def hvp(
    model: nn.Module,
    loss_fn: Callable,
    inputs: Tensor,
    targets: Tensor,
    vector: list[Tensor],
    damping: float = 0.01,
) -> list[Tensor]:
    """Hessian-vector product: (H + λI) @ v using double backprop.

    Computes H @ v via a second-order automatic differentiation trick:
        grad( grad_loss · v )

    A damping term λI is added for numerical stability.

    Args:
        model: The neural network.
        loss_fn: Callable(model, inputs, targets) -> scalar loss.
        inputs: Input tensor.
        targets: Target tensor.
        vector: List of tensors with same structure as model parameters.
        damping: Regularisation strength λ added as λI to the Hessian.

    Returns:
        List of tensors (H + λI) @ v, same structure as ``vector``.
    """
    params = [p for p in model.parameters() if p.requires_grad]

    # First-order gradients with graph retained
    loss = loss_fn(model, inputs, targets)
    first_grads = torch.autograd.grad(
        loss,
        params,
        create_graph=True,
        allow_unused=True,
    )

    # Replace None grads with zeros (allow_unused may yield None)
    first_grads_clean: list[Tensor] = []
    for g, p in zip(first_grads, params):
        if g is None:
            first_grads_clean.append(torch.zeros_like(p))
        else:
            first_grads_clean.append(g)

    # Dot product: grad · v  (scalar)
    grad_dot_v = sum((g * v).sum() for g, v in zip(first_grads_clean, vector))

    # Second-order gradients
    second_grads = torch.autograd.grad(
        grad_dot_v,
        params,
        allow_unused=True,
    )

    result: list[Tensor] = []
    for sg, v in zip(second_grads, vector):
        hv_i = torch.zeros_like(v) if sg is None else sg.detach()
        result.append(hv_i + damping * v)

    return result


# ---------------------------------------------------------------------------
# LiSSA inverse HVP
# ---------------------------------------------------------------------------


def lissa_inverse_hvp(
    model: nn.Module,
    loss_fn: Callable,
    train_inputs: list[Tensor],
    train_targets: list[Tensor],
    vector: list[Tensor],
    config: InfluenceConfig,
) -> list[Tensor]:
    """LiSSA approximation of H^{-1} @ vector via recursive sampling.

    Implements the recursion:
        h_{t+1} = v + (I - scale * H_t) h_t

    where H_t is the Hessian evaluated on a random training mini-batch, and
    ``scale`` controls convergence / damping.

    Args:
        model: The neural network.
        loss_fn: Callable(model, inputs, targets) -> scalar loss.
        train_inputs: List of training input batches (one per data point / mini-batch).
        train_targets: List of training target batches.
        vector: The vector to multiply H^{-1} by (same structure as params).
        config: InfluenceConfig specifying recursion_depth, recursion_scale, damping.

    Returns:
        Approximation of H^{-1} @ vector as a list of tensors.
    """
    n_batches = len(train_inputs)
    if n_batches == 0:
        return [v.clone() for v in vector]

    # Initialise h = v (copy)
    h: list[Tensor] = [v.clone() for v in vector]

    for _ in range(config.n_recursion_depth):
        # Pick a random batch
        idx = random.randint(0, n_batches - 1)  # noqa: S311
        batch_in = train_inputs[idx]
        batch_tgt = train_targets[idx]

        # (H + λI) @ h
        hvp_h = hvp(
            model,
            loss_fn,
            batch_in,
            batch_tgt,
            h,
            damping=config.damping,
        )

        # h = v + h - scale * (H + λI) @ h
        h = [
            v_i + h_i - config.recursion_scale * hvp_h_i
            for v_i, h_i, hvp_h_i in zip(vector, h, hvp_h)
        ]

    return h


# ---------------------------------------------------------------------------
# Influence score computation
# ---------------------------------------------------------------------------


def compute_influence_scores(
    model: nn.Module,
    loss_fn: Callable,
    train_dataset: list[tuple[Tensor, Tensor]],
    test_inputs: Tensor,
    test_targets: Tensor,
    config: InfluenceConfig,
) -> Tensor:
    """Compute influence score of each training example on test loss.

    Influence(z_i, z_test) ≈ -grad_test^T H^{-1} grad_train_i

    A positive score means the training example increases the test loss
    (harmful); a negative score means it decreases it (helpful).

    Args:
        model: The neural network (in eval mode recommended).
        loss_fn: Callable(model, inputs, targets) -> scalar loss.
        train_dataset: List of (inputs, targets) pairs — one per training point.
        test_inputs: Test input tensor.
        test_targets: Test target tensor.
        config: InfluenceConfig.

    Returns:
        1-D tensor of shape (n_train,) with influence scores.
    """
    n_train = len(train_dataset)
    params = [p for p in model.parameters() if p.requires_grad]

    # Gradient w.r.t. test point
    test_grad = compute_grad(model, loss_fn, test_inputs, test_targets, params)

    # Build list of training inputs / targets for LiSSA
    all_train_inputs = [x for x, _ in train_dataset]
    all_train_targets = [y for _, y in train_dataset]

    # Sample a subset for the LiSSA Hessian estimate if n_samples < n_train
    if config.n_samples < n_train:
        indices = list(range(n_train))
        random.shuffle(indices)
        lissa_indices = indices[: config.n_samples]
        lissa_inputs = [all_train_inputs[i] for i in lissa_indices]
        lissa_targets = [all_train_targets[i] for i in lissa_indices]
    else:
        lissa_inputs = all_train_inputs
        lissa_targets = all_train_targets

    # H^{-1} grad_test
    inv_hvp = lissa_inverse_hvp(
        model,
        loss_fn,
        lissa_inputs,
        lissa_targets,
        test_grad,
        config,
    )

    # Compute per-training-example influence scores
    scores = torch.zeros(n_train)
    for i, (tr_in, tr_tgt) in enumerate(train_dataset):
        tr_grad = compute_grad(model, loss_fn, tr_in, tr_tgt, params)
        # dot product: -grad_test^T H^{-1} grad_train_i
        dot = sum((ihvp_j * tg_j).sum() for ihvp_j, tg_j in zip(inv_hvp, tr_grad))
        scores[i] = -dot.item()

    return scores


# ---------------------------------------------------------------------------
# Top-k selection
# ---------------------------------------------------------------------------


def top_influential_examples(
    influence_scores: Tensor,
    k: int,
    mode: str = "harmful",
) -> tuple[Tensor, Tensor]:
    """Return (indices, scores) of top-k most harmful or helpful examples.

    Args:
        influence_scores: 1-D tensor of influence scores (n_train,).
        k: Number of examples to return.
        mode: ``"harmful"`` returns the k highest (most positive) scores;
              ``"helpful"`` returns the k lowest (most negative) scores.

    Returns:
        Tuple of (indices, scores), both 1-D tensors of length k.
    """
    k = min(k, influence_scores.numel())
    if mode == "harmful":
        values, indices = torch.topk(influence_scores, k, largest=True, sorted=True)
    elif mode == "helpful":
        values, indices = torch.topk(influence_scores, k, largest=False, sorted=True)
    else:
        raise ValueError(f"mode must be 'harmful' or 'helpful', got {mode!r}")
    return indices, values
