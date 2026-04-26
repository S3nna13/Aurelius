"""Test-Time Adaptation (TTA) for distribution shift robustness.

Adapts model parameters at inference time using only test inputs
(no ground truth labels). Methods: TENT (entropy minimization),
MEMO (marginal entropy over augmentations), and TTT (auxiliary task).

References:
    Wang et al. 2021 (TENT) — https://arxiv.org/abs/2006.10726
    Zhang et al. 2022 (MEMO) — https://arxiv.org/abs/2110.09506
"""

from __future__ import annotations

import logging
from collections.abc import Callable

import torch
import torch.nn as nn
from torch import Tensor
from torch.optim import Adam

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# AdaptableParams
# ---------------------------------------------------------------------------


class AdaptableParams:
    """Identifies which parameters to adapt during TTA.

    Args:
        model: The model to adapt.
        adapt_mode: One of 'affine', 'all', or 'last_layer'.
            - 'affine': only LayerNorm/BatchNorm1d affine params (weight + bias)
            - 'all': all params with requires_grad=True
            - 'last_layer': only params of the last nn.Linear found in model
    """

    def __init__(self, model: nn.Module, adapt_mode: str = "affine") -> None:
        if adapt_mode not in ("affine", "all", "last_layer"):
            raise ValueError(
                f"adapt_mode must be 'affine', 'all', or 'last_layer'; got {adapt_mode!r}"
            )
        self.model = model
        self.adapt_mode = adapt_mode

    def get_adapt_params(self) -> list[nn.Parameter]:
        """Return parameters to adapt based on adapt_mode."""
        if self.adapt_mode == "affine":
            params: list[nn.Parameter] = []
            for module in self.model.modules():
                if isinstance(module, (nn.LayerNorm, nn.BatchNorm1d)):
                    if module.weight is not None:
                        params.append(module.weight)
                    if module.bias is not None:
                        params.append(module.bias)
            return params

        elif self.adapt_mode == "all":
            return [p for p in self.model.parameters() if p.requires_grad]

        else:  # 'last_layer'
            last_linear: nn.Linear | None = None
            for module in self.model.modules():
                if isinstance(module, nn.Linear):
                    last_linear = module
            if last_linear is None:
                return []
            params = []
            if last_linear.weight is not None:
                params.append(last_linear.weight)
            if last_linear.bias is not None:
                params.append(last_linear.bias)
            return params

    def freeze_non_adapt(self) -> None:
        """Set requires_grad=False for all params NOT in get_adapt_params()."""
        adapt_set = set(id(p) for p in self.get_adapt_params())
        for p in self.model.parameters():
            if id(p) not in adapt_set:
                p.requires_grad_(False)

    def restore_all_grad(self) -> None:
        """Set requires_grad=True for all params."""
        for p in self.model.parameters():
            p.requires_grad_(True)


# ---------------------------------------------------------------------------
# EntropyMinimizer  (TENT)
# ---------------------------------------------------------------------------


class EntropyMinimizer:
    """Core TTA via entropy minimization (TENT).

    Args:
        model_fn: Callable that takes token ids (B, T) and returns logits (B, T, V).
        adapt_params: Parameters to update during adaptation.
        lr: Learning rate for the Adam optimizer.
        n_steps: Number of gradient steps per call to adapt_step.
    """

    def __init__(
        self,
        model_fn: Callable,
        adapt_params: list[nn.Parameter],
        lr: float = 0.001,
        n_steps: int = 1,
    ) -> None:
        self.model_fn = model_fn
        self.adapt_params = adapt_params
        self.lr = lr
        self.n_steps = n_steps
        self.optimizer = Adam(adapt_params, lr=lr)

    def entropy(self, logits: Tensor) -> Tensor:
        """Compute per-sample Shannon entropy.

        Args:
            logits: (B, V) raw logits.

        Returns:
            Tensor of shape (B,) with per-sample entropy values.
        """
        probs = torch.softmax(logits, dim=-1)
        return -(probs * torch.log(probs + 1e-10)).sum(dim=-1)

    def adapt_step(self, x: Tensor) -> tuple[Tensor, float]:
        """Run one (or n_steps) adaptation step(s) minimizing prediction entropy.

        Args:
            x: (B, T) input token ids.

        Returns:
            Tuple of (adapted_logits, initial_entropy_value) where:
                - adapted_logits: (B, V) logits at the last token position after adaptation
                - initial_entropy_value: mean entropy before any gradient update (float)
        """
        # Compute initial entropy before any updates (no grad)
        with torch.no_grad():
            init_logits_full = self.model_fn(x)  # (B, T, V)
            init_logits = init_logits_full[:, -1, :]  # (B, V)
            initial_entropy = self.entropy(init_logits).mean().item()

        for _ in range(self.n_steps):
            self.optimizer.zero_grad()
            logits_full = self.model_fn(x)  # (B, T, V)
            logits = logits_full[:, -1, :]  # (B, V)
            loss = self.entropy(logits).mean()
            loss.backward()
            self.optimizer.step()

        # Return final logits (no grad needed)
        with torch.no_grad():
            final_logits_full = self.model_fn(x)
            final_logits = final_logits_full[:, -1, :]

        return final_logits, initial_entropy


# ---------------------------------------------------------------------------
# AugmentationPool  (for MEMO)
# ---------------------------------------------------------------------------


class AugmentationPool:
    """Generates augmented versions of token input for MEMO.

    Args:
        n_augmentations: Number of augmented copies to produce.
        noise_scale: Probability of replacing each token with a random token.
    """

    def __init__(self, n_augmentations: int = 8, noise_scale: float = 0.1) -> None:
        self.n_augmentations = n_augmentations
        self.noise_scale = noise_scale

    def augment_token_ids(self, token_ids: Tensor, vocab_size: int) -> Tensor:
        """Produce augmented copies of a single-example token sequence.

        Args:
            token_ids: (1, T) integer tensor of token ids.
            vocab_size: Size of the vocabulary; random replacements drawn from
                [0, vocab_size).

        Returns:
            Tensor of shape (n_augmentations, T).
        """
        B, T = token_ids.shape
        assert B == 1, "augment_token_ids expects a single example (B=1)"  # noqa: S101

        # Repeat the single example n_augmentations times: (N, T)
        repeated = token_ids.expand(self.n_augmentations, T).clone()

        # For each position, replace with probability noise_scale
        mask = torch.rand(self.n_augmentations, T, device=token_ids.device) < self.noise_scale
        random_tokens = torch.randint(
            0, vocab_size, (self.n_augmentations, T), device=token_ids.device
        )
        repeated[mask] = random_tokens[mask]

        return repeated


# ---------------------------------------------------------------------------
# MEMOAdapter  (MEMO)
# ---------------------------------------------------------------------------


class MEMOAdapter:
    """MEMO: minimize marginal entropy over augmentations.

    Args:
        model_fn: Callable (B, T) -> (B, T, V) logits.
        adapt_params: Parameters to update.
        aug_pool: AugmentationPool instance.
        vocab_size: Vocabulary size.
        lr: Learning rate for Adam optimizer.
    """

    def __init__(
        self,
        model_fn: Callable,
        adapt_params: list[nn.Parameter],
        aug_pool: AugmentationPool,
        vocab_size: int,
        lr: float = 0.001,
    ) -> None:
        self.model_fn = model_fn
        self.adapt_params = adapt_params
        self.aug_pool = aug_pool
        self.vocab_size = vocab_size
        self.optimizer = Adam(adapt_params, lr=lr)

    def marginal_entropy(self, logits_batch: Tensor) -> Tensor:
        """Entropy of the mean softmax prediction across augmentations.

        Args:
            logits_batch: (N_aug, V) logits, one per augmented view.

        Returns:
            Scalar tensor: entropy of the marginal prediction.
        """
        probs = torch.softmax(logits_batch, dim=-1)  # (N_aug, V)
        marginal = probs.mean(dim=0)  # (V,)
        h = -(marginal * torch.log(marginal + 1e-10)).sum()
        return h

    def adapt_step(self, x: Tensor) -> dict[str, float]:
        """Run one MEMO adaptation step.

        Args:
            x: (1, T) input token ids (single example).

        Returns:
            Dict with keys 'marginal_entropy' (float) and 'n_augmentations' (int).
        """
        # Generate augmented views: (N_aug, T)
        aug_ids = self.aug_pool.augment_token_ids(x, self.vocab_size)

        self.optimizer.zero_grad()
        logits_full = self.model_fn(aug_ids)  # (N_aug, T, V)
        logits_last = logits_full[:, -1, :]  # (N_aug, V)

        loss = self.marginal_entropy(logits_last)
        loss.backward()
        self.optimizer.step()

        return {
            "marginal_entropy": loss.item(),
            "n_augmentations": self.aug_pool.n_augmentations,
        }
