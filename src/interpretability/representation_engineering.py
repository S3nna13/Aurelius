"""
src/interpretability/representation_engineering.py

Representation Engineering (RepE) for transformer models.

Reference: Zou et al., 2023 — "Representation Engineering: A Top-Down Approach
to AI Transparency".

The key idea: extract concept directions from contrastive activation pairs
(e.g., "honest" vs "deceptive" responses), then use these directions to steer
or analyze model behavior by manipulating the residual stream.

Pure PyTorch — no HuggingFace, no sklearn, no scipy.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class RepEngConfig:
    """Configuration for Representation Engineering.

    Attributes
    ----------
    layer_idx   : Which transformer layer to extract representations from.
                  -1 means the last layer.
    batch_size  : Batch size for processing examples.
    n_directions: Number of PCA directions to extract (currently used as metadata).
    normalize   : Whether to normalize steering vectors to unit norm.
    """

    layer_idx: int = -1
    batch_size: int = 8
    n_directions: int = 1
    normalize: bool = True


# ---------------------------------------------------------------------------
# RepresentationExtractor
# ---------------------------------------------------------------------------


class RepresentationExtractor:
    """Extract hidden representations from a specific transformer layer.

    Uses forward hooks to capture intermediate activations without modifying
    the model.

    Parameters
    ----------
    model     : An AureliusTransformer-like model with a ``layers`` ModuleList.
    layer_idx : The layer index to extract from.  Negative indices (e.g. -1
                for the last layer) are resolved against ``len(model.layers)``.
    """

    def __init__(self, model: nn.Module, layer_idx: int) -> None:
        self.model = model
        n_layers = len(model.layers)  # type: ignore[arg-type]
        # Resolve negative index once so hook attachment is unambiguous
        if layer_idx < 0:
            layer_idx = n_layers + layer_idx
        if not (0 <= layer_idx < n_layers):
            raise IndexError(f"layer_idx {layer_idx} out of range for model with {n_layers} layers")
        self.layer_idx = layer_idx

    def extract(self, input_ids: Tensor) -> Tensor:
        """Extract representations at the configured layer.

        Registers a temporary forward hook on the target layer, runs a
        no-grad forward pass, and returns the captured hidden states.

        Parameters
        ----------
        input_ids : (B, T) token indices.

        Returns
        -------
        Tensor of shape (B, T, d_model) — hidden states after the chosen
        transformer block (post-residual, pre-next-layer-norm).
        """
        captured: dict[int, Tensor] = {}
        layer = self.model.layers[self.layer_idx]  # type: ignore[index]

        def _hook(module: nn.Module, inputs, output) -> None:
            # TransformerBlock returns (hidden, kv_tuple); handle both cases
            act = output[0] if isinstance(output, tuple) else output
            captured[self.layer_idx] = act.detach()

        handle = layer.register_forward_hook(_hook)
        try:
            with torch.no_grad():
                self.model(input_ids)
        finally:
            handle.remove()

        if self.layer_idx not in captured:
            raise RuntimeError(f"Hook did not fire for layer {self.layer_idx}")

        return captured[self.layer_idx]


# ---------------------------------------------------------------------------
# extract_concept_direction
# ---------------------------------------------------------------------------


def extract_concept_direction(
    model: nn.Module,
    positive_ids: Tensor,  # (N, T) token ids for positive examples
    negative_ids: Tensor,  # (N, T) token ids for negative examples
    layer_idx: int = -1,
    normalize: bool = True,
) -> Tensor:
    """Extract a concept direction using contrastive activation pairs.

    Computes ``mean(positive_hiddens[:, -1, :]) - mean(negative_hiddens[:, -1, :])``
    where the mean is taken over the last token position of each sequence,
    then averaged over the batch.

    Parameters
    ----------
    model        : AureliusTransformer-compatible model.
    positive_ids : (N, T) token ids for the positive concept class.
    negative_ids : (N, T) token ids for the negative concept class.
    layer_idx    : Which layer to extract from.  -1 = last layer.
    normalize    : If True, normalize the output to unit norm.

    Returns
    -------
    Tensor of shape (d_model,).
    """
    extractor = RepresentationExtractor(model, layer_idx)

    pos_hiddens = extractor.extract(positive_ids)  # (N, T, D)
    neg_hiddens = extractor.extract(negative_ids)  # (N, T, D)

    # Last-token representation, averaged over the batch
    pos_mean = pos_hiddens[:, -1, :].mean(dim=0)  # (D,)
    neg_mean = neg_hiddens[:, -1, :].mean(dim=0)  # (D,)

    direction = pos_mean - neg_mean  # (D,)

    if normalize:
        norm = direction.norm()
        if norm > 1e-8:
            direction = direction / norm

    return direction


# ---------------------------------------------------------------------------
# SteeringVectorBank
# ---------------------------------------------------------------------------


class SteeringVectorBank:
    """Persistent store of named steering vectors with composition utilities.

    All vectors in the bank are plain (d_model,) tensors — no layer-index
    metadata is stored here; callers are responsible for tracking which
    layer a vector was extracted from.

    Usage
    -----
    >>> bank = SteeringVectorBank()
    >>> bank.add("honesty", direction_tensor)
    >>> composed = bank.compose(["honesty", "helpfulness"])
    """

    def __init__(self) -> None:
        self._bank: dict[str, Tensor] = {}

    # ------------------------------------------------------------------
    # Basic CRUD
    # ------------------------------------------------------------------

    def add(self, name: str, vector: Tensor) -> None:
        """Add or overwrite a steering vector by *name*."""
        self._bank[name] = vector

    def get(self, name: str) -> Tensor:
        """Retrieve a steering vector by *name*.

        Raises
        ------
        KeyError if *name* is not in the bank.
        """
        if name not in self._bank:
            raise KeyError(f"No steering vector named '{name}'.  Available: {list(self._bank)}")
        return self._bank[name]

    def remove(self, name: str) -> None:
        """Remove the steering vector stored under *name*.

        Raises
        ------
        KeyError if *name* is not in the bank.
        """
        if name not in self._bank:
            raise KeyError(f"No steering vector named '{name}'.  Available: {list(self._bank)}")
        del self._bank[name]

    def list_names(self) -> list[str]:
        """Return a list of all stored vector names."""
        return list(self._bank.keys())

    # ------------------------------------------------------------------
    # Composition utilities
    # ------------------------------------------------------------------

    def compose(
        self,
        names: list[str],
        weights: list[float] | None = None,
    ) -> Tensor:
        """Compose multiple vectors as a weighted sum.

        If *weights* is None, all vectors are summed with equal weight and
        the result is normalized to unit norm.  If *weights* is provided,
        the vectors are combined as ``sum(w_i * v_i)`` without re-normalizing.

        Parameters
        ----------
        names   : Names of vectors to compose (must all be present in the bank).
        weights : Optional per-vector weights.  Must have the same length as
                  *names*.  If None, uses equal weights and normalizes.

        Returns
        -------
        (d_model,) composed tensor.
        """
        if not names:
            raise ValueError("names must not be empty")

        vectors = [self.get(n) for n in names]

        if weights is None:
            # Equal-weight sum, then normalize
            composed = torch.stack(vectors, dim=0).mean(dim=0)  # (D,)
            norm = composed.norm()
            if norm > 1e-8:
                composed = composed / norm
        else:
            if len(weights) != len(vectors):
                raise ValueError(
                    f"len(weights) ({len(weights)}) must equal len(names) ({len(names)})"
                )
            weight_t = torch.tensor(weights, dtype=vectors[0].dtype, device=vectors[0].device)
            stacked = torch.stack(vectors, dim=0)  # (N, D)
            composed = (weight_t.unsqueeze(1) * stacked).sum(dim=0)  # (D,)

        return composed

    def project_out(self, vector: Tensor, direction: str) -> Tensor:
        """Remove the component of *vector* along the named stored direction.

        Implements the Gram-Schmidt step:
            v_projected = v - (v · d / |d|²) * d

        where *d* is the stored direction retrieved by *direction*.

        Parameters
        ----------
        vector    : (d_model,) tensor to project.
        direction : Name of the direction to remove from *vector*.

        Returns
        -------
        (d_model,) tensor with the named direction component removed.
        """
        d = self.get(direction)  # (D,)
        d_sq_norm = d.dot(d)
        if d_sq_norm < 1e-16:
            return vector.clone()
        coeff = vector.dot(d) / d_sq_norm
        return vector - coeff * d

    def interpolate(self, name_a: str, name_b: str, alpha: float) -> Tensor:
        """Linearly interpolate between two stored vectors.

        Returns ``alpha * vec_a + (1 - alpha) * vec_b``.

        Parameters
        ----------
        name_a : Name of the first vector (weight = alpha).
        name_b : Name of the second vector (weight = 1 - alpha).
        alpha  : Interpolation coefficient in [0, 1].
                 alpha=1 returns vec_a; alpha=0 returns vec_b.

        Returns
        -------
        (d_model,) interpolated tensor.
        """
        vec_a = self.get(name_a)
        vec_b = self.get(name_b)
        return alpha * vec_a + (1.0 - alpha) * vec_b


# ---------------------------------------------------------------------------
# apply_steering_hook
# ---------------------------------------------------------------------------


def apply_steering_hook(
    model: nn.Module,
    steering_vector: Tensor,  # (d_model,)
    layer_idx: int,
    scale: float = 1.0,
) -> Callable:
    """Register a forward hook that adds ``scale * steering_vector`` to
    the hidden states at *layer_idx*.

    The hook is attached to ``model.layers[layer_idx]`` and fires after the
    TransformerBlock's forward pass, adding the (broadcast) steering vector
    to the residual-stream tensor.

    Parameters
    ----------
    model          : AureliusTransformer-compatible model.
    steering_vector: (d_model,) direction to add.
    layer_idx      : Index of the layer to steer (negative indices supported).
    scale          : Multiplicative coefficient for the steering vector.

    Returns
    -------
    The hook handle returned by ``register_forward_hook``.
    Call ``handle.remove()`` to unregister the hook.
    """
    n_layers = len(model.layers)  # type: ignore[arg-type]
    # Resolve negative index
    resolved_idx = layer_idx if layer_idx >= 0 else n_layers + layer_idx
    if not (0 <= resolved_idx < n_layers):
        raise IndexError(f"layer_idx {layer_idx} out of range for model with {n_layers} layers")

    layer = model.layers[resolved_idx]  # type: ignore[index]

    def _steering_hook(module: nn.Module, inputs, output):
        is_tuple = isinstance(output, tuple)
        act = output[0] if is_tuple else output  # (B, T, D)
        steered = act + scale * steering_vector.to(act.device)
        return (steered,) + output[1:] if is_tuple else steered

    handle = layer.register_forward_hook(_steering_hook)
    return handle
