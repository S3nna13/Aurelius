"""Aurelius -- Activation Steering / Representation Engineering.

Implements steering vectors (Zou et al. 2023, Turner et al. 2023): extract a
"steering vector" (e.g. mean-diff or PCA direction) from contrastive positive /
negative activations, then add alpha * steering_vector to the residual stream
at a given layer during generation.  This shifts model behaviour without
fine-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Config / dataclasses
# ---------------------------------------------------------------------------


@dataclass
class SteeringConfig:
    """Configuration for activation steering.

    Attributes:
        layer_indices: Which transformer layers to steer.
        coefficient: Steering strength (negative value negates the concept).
        normalize: Normalize steering vectors to unit norm before storing.
        mode: "add" | "project_out" | "clamp"
        clamp_value: Threshold used when mode="clamp".
    """

    layer_indices: list[int]
    coefficient: float = 1.0
    normalize: bool = True
    mode: str = "add"  # "add" | "project_out" | "clamp"
    clamp_value: float = 5.0


@dataclass
class SteeringVector:
    """A unit steering direction to be applied at a specific layer.

    Attributes:
        direction: (d_model,) steering direction (unit norm when normalize=True).
        layer_idx: Which transformer layer to intervene on.
        concept: Human-readable name of the concept being steered.
        explained_variance: Explained variance from PCA fit; 0.0 for mean-diff.
    """

    direction: torch.Tensor  # (d_model,)
    layer_idx: int
    concept: str = "steering"
    explained_variance: float = 0.0


# ---------------------------------------------------------------------------
# Standalone extraction helpers
# ---------------------------------------------------------------------------


def extract_mean_diff_vector(
    positive_activations: torch.Tensor,  # (N_pos, D)
    negative_activations: torch.Tensor,  # (N_neg, D)
    normalize: bool = True,
) -> torch.Tensor:
    """Compute mean-difference steering vector: mean(pos) - mean(neg).

    Args:
        positive_activations: (N_pos, D) activations for positive examples.
        negative_activations: (N_neg, D) activations for negative examples.
        normalize: If True, return unit-norm vector.

    Returns:
        (D,) steering direction.
    """
    direction = positive_activations.mean(dim=0) - negative_activations.mean(dim=0)
    if normalize:
        direction = F.normalize(direction, dim=0)
    return direction


def extract_pca_vector(
    contrastive_activations: torch.Tensor,  # (2*N, D)
    normalize: bool = True,
) -> tuple[torch.Tensor, float]:
    """Extract the top PCA component from contrastive activations.

    Centers the activations, then uses torch.pca_lowrank to get the first
    principal component.

    Args:
        contrastive_activations: (2*N, D) positive and negative activations.
        normalize: If True, normalize the direction to unit norm.

    Returns:
        Tuple of ((D,) direction tensor, explained_variance_ratio float).
    """
    centered = contrastive_activations - contrastive_activations.mean(dim=0, keepdim=True)
    # pca_lowrank returns (U, S, V); first PC lives in V[:, 0]
    _U, S, V = torch.pca_lowrank(centered, q=1)
    direction = V[:, 0]  # (D,)
    if normalize:
        direction = F.normalize(direction, dim=0)

    # Explained variance ratio = S[0]^2 / sum(S_all^2)
    # With q=1 we only have S[0]; compute total variance from data directly
    total_var = (centered**2).sum()
    if total_var.item() == 0.0:
        explained_var = 0.0
    else:
        explained_var = float((S[0] ** 2) / total_var)
        # Clamp to (0, 1] for safety
        explained_var = max(min(explained_var, 1.0), 1e-9)

    return direction, explained_var


# ---------------------------------------------------------------------------
# collect_activations
# ---------------------------------------------------------------------------


def collect_activations(
    model: nn.Module,
    input_ids: torch.Tensor,  # (B, T)
    layer_idx: int,
) -> torch.Tensor:
    """Collect mean-pooled activations from layer ``layer_idx``.

    Registers a single forward hook on ``model.layers[layer_idx]``, runs the
    forward pass, and returns the (B, D) mean-pooled hidden states.

    The TransformerBlock returns ``(hidden, kv)``; the hook captures
    ``hidden`` (the first element of that tuple).

    Args:
        model: AureliusTransformer (or any model with a `.layers` ModuleList).
        input_ids: (B, T) integer token ids.
        layer_idx: Which layer to hook.

    Returns:
        (B, D) mean-pooled activations.
    """
    captured: list[torch.Tensor] = []

    def _hook(module, inputs, output):
        # TransformerBlock returns (hidden, kv); grab hidden
        hidden = output[0] if isinstance(output, tuple) else output
        # hidden: (B, T, D) — mean-pool over T
        captured.append(hidden.detach().mean(dim=1))  # (B, D)

    handle = model.layers[layer_idx].register_forward_hook(_hook)
    try:
        model.eval()
        with torch.no_grad():
            model(input_ids)
    finally:
        handle.remove()

    return captured[0]  # (B, D)


# ---------------------------------------------------------------------------
# SteeringHook  (context manager for a single layer)
# ---------------------------------------------------------------------------


class SteeringHook:
    """Context manager that injects a steering vector into one layer's output.

    The hook modifies the (B, T, D) hidden state tensor returned by the layer
    according to ``cfg.mode``:
        - "add":          hidden += coeff * direction
        - "project_out":  remove the component along ``direction``
        - "clamp":        clamp activations projected onto direction

    Args:
        model: AureliusTransformer (or compatible).
        steering_vector: The SteeringVector to apply.
        cfg: SteeringConfig controlling strength and mode.
    """

    def __init__(
        self,
        model: nn.Module,
        steering_vector: SteeringVector,
        cfg: SteeringConfig,
    ) -> None:
        self.model = model
        self.steering_vector = steering_vector
        self.cfg = cfg
        self._handle = None

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> SteeringHook:
        layer_idx = self.steering_vector.layer_idx

        def _hook(module, inputs, output):
            if isinstance(output, tuple):
                hidden, *rest = output
                hidden = self._apply_steering(hidden)
                return (hidden, *rest)
            else:
                return self._apply_steering(output)

        self._handle = self.model.layers[layer_idx].register_forward_hook(_hook)
        return self

    def __exit__(self, *args) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    # ------------------------------------------------------------------
    # Core steering logic
    # ------------------------------------------------------------------

    def _apply_steering(self, output: torch.Tensor) -> torch.Tensor:
        """Apply steering to a (B, T, D) hidden state tensor.

        Args:
            output: (B, T, D) hidden states from the layer.

        Returns:
            Modified (B, T, D) hidden states.
        """
        direction = self.steering_vector.direction.to(output.device)  # (D,)
        coeff = self.cfg.coefficient
        mode = self.cfg.mode

        if mode == "add":
            # Broadcast direction (D,) over (B, T, D)
            output = output + coeff * direction

        elif mode == "project_out":
            # Remove the component along direction from every token position
            # proj shape: (B, T, 1) * (D,) -> (B, T, D)
            proj_scalar = output @ direction.unsqueeze(-1)  # (B, T, 1)
            proj = proj_scalar * direction  # (B, T, D)
            output = output - proj

        elif mode == "clamp":
            # Clamp activations in the direction to [-clamp_value, clamp_value]
            proj_scalar = output @ direction  # (B, T)
            clamped = proj_scalar.clamp(-self.cfg.clamp_value, self.cfg.clamp_value)
            diff = (clamped - proj_scalar).unsqueeze(-1)  # (B, T, 1)
            output = output + diff * direction  # adjust in-direction component

        else:
            raise ValueError(f"Unknown steering mode: {mode!r}")

        return output


# ---------------------------------------------------------------------------
# ActivationSteerer  (high-level API)
# ---------------------------------------------------------------------------


class ActivationSteerer:
    """Extract steering vectors and apply them at inference time.

    Args:
        model: AureliusTransformer (or compatible nn.Module with ``.layers``).
        cfg: SteeringConfig specifying layers, coefficient, and mode.
    """

    def __init__(self, model: nn.Module, cfg: SteeringConfig) -> None:
        self.model = model
        self.cfg = cfg

    # ------------------------------------------------------------------
    # Fitting
    # ------------------------------------------------------------------

    def fit_from_pairs(
        self,
        positive_ids: torch.Tensor,  # (N, T)
        negative_ids: torch.Tensor,  # (N, T)
        method: str = "mean_diff",  # "mean_diff" | "pca"
    ) -> list[SteeringVector]:
        """Extract steering vectors for each configured layer.

        For each layer index in ``cfg.layer_indices``:
            1. Collect mean-pooled activations for positive / negative examples.
            2. Compute the steering direction via ``method``.

        Args:
            positive_ids: (N, T) token ids for positive examples.
            negative_ids: (N, T) token ids for negative examples.
            method: "mean_diff" or "pca".

        Returns:
            List of SteeringVector, one per layer in cfg.layer_indices.
        """
        vectors: list[SteeringVector] = []

        for layer_idx in self.cfg.layer_indices:
            pos_acts = collect_activations(self.model, positive_ids, layer_idx)  # (N, D)
            neg_acts = collect_activations(self.model, negative_ids, layer_idx)  # (N, D)

            if method == "mean_diff":
                direction = extract_mean_diff_vector(
                    pos_acts, neg_acts, normalize=self.cfg.normalize
                )
                explained_var = 0.0
            elif method == "pca":
                contrastive = torch.cat([pos_acts, neg_acts], dim=0)  # (2N, D)
                direction, explained_var = extract_pca_vector(
                    contrastive, normalize=self.cfg.normalize
                )
            else:
                raise ValueError(f"Unknown method: {method!r}")

            vectors.append(
                SteeringVector(
                    direction=direction,
                    layer_idx=layer_idx,
                    concept="steering",
                    explained_variance=explained_var,
                )
            )

        return vectors

    # ------------------------------------------------------------------
    # Steered generation
    # ------------------------------------------------------------------

    def steer(
        self,
        input_ids: torch.Tensor,  # (1, T) or (B, T)
        steering_vectors: list[SteeringVector],
        max_new_tokens: int = 10,
    ) -> torch.Tensor:
        """Greedy-decode with all steering vectors applied simultaneously.

        Opens one SteeringHook per steering vector, then greedily decodes
        ``max_new_tokens`` tokens.

        Args:
            input_ids: (1, T) starting token ids (batch size 1 expected).
            steering_vectors: SteeringVectors from :meth:`fit_from_pairs`.
            max_new_tokens: Number of tokens to generate.

        Returns:
            (1, max_new_tokens) generated token ids.
        """
        self.model.eval()

        # Build list of context managers
        hooks = [SteeringHook(self.model, sv, self.cfg) for sv in steering_vectors]

        generated: list[torch.Tensor] = []
        cur_ids = input_ids  # (1, T)

        # Enter all hooks
        [h.__enter__() for h in hooks]
        try:
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    _loss, logits, _pkv = self.model(cur_ids)
                    # Greedy: pick the argmax at the last position
                    next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)  # (1, 1)
                    generated.append(next_token)
                    cur_ids = torch.cat([cur_ids, next_token], dim=1)
        finally:
            for h in hooks:
                h.__exit__(None, None, None)

        return torch.cat(generated, dim=1)  # (1, max_new_tokens)

    # ------------------------------------------------------------------
    # Probing
    # ------------------------------------------------------------------

    def measure_concept_activation(
        self,
        input_ids: torch.Tensor,
        steering_vector: SteeringVector,
    ) -> float:
        """Measure how strongly the concept is expressed in the activations.

        Projects the layer activations onto the steering direction and returns
        the mean dot product across batch and sequence positions.

        Args:
            input_ids: (B, T) token ids.
            steering_vector: The SteeringVector to probe.

        Returns:
            Mean projection (float).
        """
        acts = collect_activations(self.model, input_ids, steering_vector.layer_idx)  # (B, D)
        direction = steering_vector.direction.to(acts.device)  # (D,)
        return float((acts @ direction).mean().item())
