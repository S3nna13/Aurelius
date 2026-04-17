"""Activation Steering / Concept Vectors (Turner et al. 2023, Zou et al. 2023).

Extract concept directions from activation differences between contrastive pairs,
then steer model behavior by adding scaled concept vectors during inference.

Pure PyTorch only — no transformers, einops, or other external deps.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# ContrastivePairCollector
# ---------------------------------------------------------------------------

class ContrastivePairCollector:
    """Collect activations from positive/negative concept pairs.

    Stores mean-pooled (B, D) representations from (B, T, D) activation tensors.
    """

    def __init__(self) -> None:
        self.pos_list: list[Tensor] = []
        self.neg_list: list[Tensor] = []

    def record(self, pos_activations: Tensor, neg_activations: Tensor) -> None:
        """Store a contrastive pair.

        Args:
            pos_activations: (B, T, D) activations for the positive concept.
            neg_activations: (B, T, D) activations for the negative concept.
        """
        # Mean-pool over T → (B, D)
        pos_pooled = pos_activations.mean(dim=1)  # (B, D)
        neg_pooled = neg_activations.mean(dim=1)  # (B, D)
        self.pos_list.append(pos_pooled)
        self.neg_list.append(neg_pooled)

    def get_arrays(self) -> tuple[Tensor, Tensor]:
        """Return stacked (N, D) tensors for positive and negative examples.

        Returns:
            Tuple of (pos_matrix, neg_matrix) each of shape (N, D).
        """
        pos_matrix = torch.cat(self.pos_list, dim=0)  # (N, D)
        neg_matrix = torch.cat(self.neg_list, dim=0)  # (N, D)
        return pos_matrix, neg_matrix

    def reset(self) -> None:
        """Clear all stored activations."""
        self.pos_list = []
        self.neg_list = []


# ---------------------------------------------------------------------------
# ConceptVectorExtractor
# ---------------------------------------------------------------------------

class ConceptVectorExtractor:
    """Extract a concept direction from contrastive activation pairs.

    Supported methods:
        "mean_diff" — normalize(mean(pos) - mean(neg))
        "pca"       — first principal component of (pos - neg) via SVD
        "logistic"  — weight vector of a logistic regression trained on pos/neg
    """

    def __init__(self, method: str = "mean_diff") -> None:
        if method not in ("mean_diff", "pca", "logistic"):
            raise ValueError(f"Unknown method: {method!r}. Choose 'mean_diff', 'pca', or 'logistic'.")
        self.method = method

    # ------------------------------------------------------------------
    # Individual extraction methods
    # ------------------------------------------------------------------

    def mean_diff(self, pos: Tensor, neg: Tensor) -> Tensor:
        """Mean-difference direction, L2-normalised.

        Args:
            pos: (N, D) positive activations.
            neg: (N, D) negative activations.

        Returns:
            Unit vector of shape (D,).
        """
        diff = pos.mean(dim=0) - neg.mean(dim=0)  # (D,)
        return F.normalize(diff, dim=0)

    def pca_direction(self, pos: Tensor, neg: Tensor) -> Tensor:
        """First principal component of (pos - neg), L2-normalised.

        Args:
            pos: (N, D) positive activations.
            neg: (N, D) negative activations.

        Returns:
            Unit vector of shape (D,).
        """
        diffs = pos - neg  # (N, D)
        # Centre the differences
        diffs = diffs - diffs.mean(dim=0, keepdim=True)
        # SVD: diffs = U @ S @ Vh; first right singular vector = first PC
        _, _, Vh = torch.linalg.svd(diffs, full_matrices=False)
        direction = Vh[0]  # (D,)
        return F.normalize(direction, dim=0)

    def logistic_direction(self, pos: Tensor, neg: Tensor, n_steps: int = 50) -> Tensor:
        """Train logistic regression and return normalised weight vector.

        Args:
            pos: (N, D) positive activations.
            neg: (N, D) negative activations.
            n_steps: Number of SGD optimisation steps.

        Returns:
            Unit vector of shape (D,).
        """
        N_pos = pos.shape[0]
        N_neg = neg.shape[0]
        X = torch.cat([pos, neg], dim=0).float()          # (N_pos+N_neg, D)
        y = torch.cat([
            torch.ones(N_pos, 1),
            torch.zeros(N_neg, 1),
        ], dim=0)                                           # (N_pos+N_neg, 1)

        linear = nn.Linear(X.shape[1], 1, bias=True)
        nn.init.zeros_(linear.weight)
        nn.init.zeros_(linear.bias)

        optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

        for _ in range(n_steps):
            optimizer.zero_grad()
            logits = linear(X)                              # (N, 1)
            loss = F.binary_cross_entropy_with_logits(logits, y)
            loss.backward()
            optimizer.step()

        weight = linear.weight.data.squeeze(0)  # (D,)
        return F.normalize(weight, dim=0)

    # ------------------------------------------------------------------
    # Unified entry point
    # ------------------------------------------------------------------

    def extract(self, pos: Tensor, neg: Tensor) -> Tensor:
        """Extract a concept vector using the configured method.

        Args:
            pos: (N, D) positive activations.
            neg: (N, D) negative activations.

        Returns:
            Unit concept vector of shape (D,).
        """
        if self.method == "mean_diff":
            return self.mean_diff(pos, neg)
        if self.method == "pca":
            return self.pca_direction(pos, neg)
        if self.method == "logistic":
            return self.logistic_direction(pos, neg)
        raise ValueError(f"Unknown method: {self.method!r}")


# ---------------------------------------------------------------------------
# ActivationHook
# ---------------------------------------------------------------------------

class ActivationHook:
    """PyTorch forward hook that captures or modifies layer activations.

    Modes:
        "capture"     — store output in self.captured, pass through unchanged.
        "add"         — add alpha * steering_vector to output.
        "project_out" — remove the component of output along steering_vector.
    """

    def __init__(self, layer_idx: int, mode: str = "capture") -> None:
        if mode not in ("capture", "add", "project_out"):
            raise ValueError(f"Unknown mode: {mode!r}. Choose 'capture', 'add', or 'project_out'.")
        self.layer_idx = layer_idx
        self.mode = mode
        self.captured: Optional[Tensor] = None
        self.steering_vector: Optional[Tensor] = None
        self.alpha: float = 1.0
        self.handle: Optional[torch.utils.hooks.RemovableHook] = None

    def hook_fn(self, module: nn.Module, input: tuple, output: Tensor) -> Tensor:
        """Forward hook callback.

        Args:
            module: The hooked module (unused).
            input:  Module inputs (unused).
            output: Module output tensor of shape (B, T, D) or similar.

        Returns:
            Possibly modified output tensor.
        """
        if self.mode == "capture":
            self.captured = output.detach()
            return output

        if self.mode == "add":
            if self.steering_vector is None:
                raise RuntimeError("steering_vector not set; call set_steering() first.")
            v = self.steering_vector.to(output.device, output.dtype)
            return output + self.alpha * v

        if self.mode == "project_out":
            if self.steering_vector is None:
                raise RuntimeError("steering_vector not set; call set_steering() first.")
            v = self.steering_vector.to(output.device, output.dtype)
            # output - (output · v / ||v||²) * v
            v_norm_sq = (v * v).sum()
            # Broadcast: output is (..., D), v is (D,)
            proj_scalar = (output * v).sum(dim=-1, keepdim=True) / v_norm_sq
            return output - proj_scalar * v

        raise RuntimeError(f"Unknown mode: {self.mode!r}")  # should never reach here

    def set_steering(self, vector: Tensor, alpha: float) -> None:
        """Set the steering vector and scale factor.

        Args:
            vector: Concept direction of shape (D,).
            alpha:  Multiplicative scale.
        """
        self.steering_vector = vector
        self.alpha = alpha

    def register(self, module: nn.Module) -> torch.utils.hooks.RemovableHook:
        """Register the hook on a module.

        Args:
            module: The nn.Module to attach the hook to.

        Returns:
            The removable hook handle (also stored as self.handle).
        """
        self.handle = module.register_forward_hook(self.hook_fn)
        return self.handle

    def remove(self) -> None:
        """Remove the registered hook."""
        if self.handle is not None:
            self.handle.remove()
            self.handle = None


# ---------------------------------------------------------------------------
# ActivationSteerer
# ---------------------------------------------------------------------------

class ActivationSteerer:
    """Apply concept-vector steering to a model at inference time.

    The model must expose a ``.layers`` attribute of type ``nn.ModuleList``.
    """

    def __init__(self, model: nn.Module, layers: list[int]) -> None:
        if not hasattr(model, "layers"):
            raise AttributeError("model must expose a 'layers' attribute (nn.ModuleList).")
        self.model = model
        self.layer_indices = layers
        self._hooks: list[ActivationHook] = []

    def add_hooks(
        self,
        concept_vector: Tensor,
        alpha: float,
        mode: str = "add",
    ) -> list[ActivationHook]:
        """Attach steering hooks to the configured layers.

        Args:
            concept_vector: Unit concept vector of shape (D,).
            alpha:          Steering coefficient.
            mode:           Hook mode — "add" or "project_out".

        Returns:
            List of registered ActivationHook instances.
        """
        self._hooks = []
        for idx in self.layer_indices:
            hook = ActivationHook(layer_idx=idx, mode=mode)
            hook.set_steering(concept_vector, alpha)
            hook.register(self.model.layers[idx])
            self._hooks.append(hook)
        return self._hooks

    def remove_hooks(self) -> None:
        """Remove all active steering hooks."""
        for hook in self._hooks:
            hook.remove()
        self._hooks = []

    def steered_forward(
        self,
        input_ids: Tensor,
        concept_vector: Tensor,
        alpha: float,
    ) -> Tensor:
        """Run a steered forward pass.

        Registers hooks, runs the model, removes hooks, returns logits.

        Args:
            input_ids:      (B, T) token-id tensor.
            concept_vector: Unit concept vector (D,).
            alpha:          Steering coefficient.

        Returns:
            Logits tensor of shape (B, T, vocab_size).
        """
        self.add_hooks(concept_vector, alpha, mode="add")
        try:
            logits = self.model(input_ids)
        finally:
            self.remove_hooks()
        return logits


# ---------------------------------------------------------------------------
# SteeringEffect
# ---------------------------------------------------------------------------

class SteeringEffect:
    """Measure the effect of activation steering on model output distributions."""

    def __init__(self) -> None:
        pass

    def logit_diff(
        self,
        original_logits: Tensor,
        steered_logits: Tensor,
        target_token: int,
    ) -> float:
        """Difference in log-prob of target_token at the last position.

        Args:
            original_logits: (B, T, V) or (T, V) logits before steering.
            steered_logits:  (B, T, V) or (T, V) logits after steering.
            target_token:    Vocabulary index of the token of interest.

        Returns:
            Mean over batch of (log_prob_steered - log_prob_original) for target_token.
        """
        # Normalise to at least 2-D by treating (T, V) as (1, T, V)
        if original_logits.dim() == 2:
            original_logits = original_logits.unsqueeze(0)
            steered_logits = steered_logits.unsqueeze(0)

        # Last position
        orig_last = original_logits[:, -1, :]   # (B, V)
        steer_last = steered_logits[:, -1, :]   # (B, V)

        orig_lp = F.log_softmax(orig_last.float(), dim=-1)[:, target_token]   # (B,)
        steer_lp = F.log_softmax(steer_last.float(), dim=-1)[:, target_token] # (B,)

        return (steer_lp - orig_lp).mean().item()

    def kl_divergence(
        self,
        original_logits: Tensor,
        steered_logits: Tensor,
    ) -> float:
        """Mean KL(original || steered) at the last position.

        Args:
            original_logits: (B, T, V) or (T, V) logits before steering.
            steered_logits:  (B, T, V) or (T, V) logits after steering.

        Returns:
            Non-negative float; 0.0 when both distributions are identical.
        """
        if original_logits.dim() == 2:
            original_logits = original_logits.unsqueeze(0)
            steered_logits = steered_logits.unsqueeze(0)

        orig_last = original_logits[:, -1, :].float()    # (B, V)
        steer_last = steered_logits[:, -1, :].float()    # (B, V)

        # KL(P || Q) where P = original, Q = steered
        p_log = F.log_softmax(orig_last, dim=-1)   # (B, V)
        q_log = F.log_softmax(steer_last, dim=-1)  # (B, V)
        p = p_log.exp()                             # (B, V)

        # Sum over vocab, mean over batch
        kl = (p * (p_log - q_log)).sum(dim=-1).mean()
        return kl.item()

    def top_k_shift(
        self,
        original_logits: Tensor,
        steered_logits: Tensor,
        k: int = 5,
    ) -> float:
        """Jaccard similarity of top-k tokens between original and steered.

        Args:
            original_logits: (B, T, V) or (T, V) logits before steering.
            steered_logits:  (B, T, V) or (T, V) logits after steering.
            k:               Number of top tokens to compare.

        Returns:
            Float in [0, 1]; 1.0 when top-k sets are identical.
        """
        if original_logits.dim() == 2:
            original_logits = original_logits.unsqueeze(0)
            steered_logits = steered_logits.unsqueeze(0)

        orig_last = original_logits[:, -1, :]    # (B, V)
        steer_last = steered_logits[:, -1, :]    # (B, V)

        batch_size = orig_last.shape[0]
        total_jaccard = 0.0

        for b in range(batch_size):
            orig_topk = set(torch.topk(orig_last[b], k).indices.tolist())
            steer_topk = set(torch.topk(steer_last[b], k).indices.tolist())
            intersection = len(orig_topk & steer_topk)
            union = len(orig_topk | steer_topk)
            total_jaccard += intersection / union if union > 0 else 1.0

        return total_jaccard / batch_size
