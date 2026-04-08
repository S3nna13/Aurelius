"""Aurelius -- Activation Steering / Representation Engineering.

Implements steering vectors (Zou et al. 2023, Turner et al. 2023): find a
"steering vector" (difference of mean activations between positive and negative
examples), then add alpha * steering_vector to the residual stream at a given
layer during generation. This shifts model behavior toward the desired
direction without fine-tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# SteeringVector dataclass
# ---------------------------------------------------------------------------

@dataclass
class SteeringVector:
    """A steering direction to be applied at a specific layer.

    Attributes:
        direction: (d_model,) unit-normalized steering direction.
        layer_idx: Which transformer layer to intervene on.
        label: Human-readable label, e.g. "helpful", "honest".
        source: How the vector was computed ("mean_diff", "pca", "probing").
    """
    direction: torch.Tensor    # (d_model,)
    layer_idx: int
    label: str
    source: str = "mean_diff"


# ---------------------------------------------------------------------------
# SteeringVectorExtractor
# ---------------------------------------------------------------------------

class SteeringVectorExtractor:
    """Extract steering vectors from contrastive positive/negative text pairs.

    Args:
        model: AureliusTransformer instance.
        tokenizer_encode: Callable that maps str -> list[int].
        layer_idx: Which layer's hidden states to capture.
        max_seq_len: Maximum sequence length for tokenization.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer_encode: Callable[[str], list[int]],
        layer_idx: int,
        max_seq_len: int = 64,
    ) -> None:
        self.model = model
        self.tokenizer_encode = tokenizer_encode
        self.layer_idx = layer_idx
        self.max_seq_len = max_seq_len

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_hidden_states(self, texts: list[str]) -> torch.Tensor:
        """Forward pass capturing hidden states at self.layer_idx.

        Uses a forward hook on model.layers[layer_idx].
        Returns mean over sequence dimension: (len(texts), d_model).
        """
        captured: list[torch.Tensor] = []

        def hook(module, inputs, output):
            # TransformerBlock returns (hidden, kv); grab the hidden state
            if isinstance(output, tuple):
                hidden = output[0]
            else:
                hidden = output
            # hidden: (B, S, D) -- mean over seq dim
            captured.append(hidden.mean(dim=1).detach())

        handle = self.model.layers[self.layer_idx].register_forward_hook(hook)
        try:
            self.model.eval()
            with torch.no_grad():
                results: list[torch.Tensor] = []
                for text in texts:
                    token_ids = self.tokenizer_encode(text)
                    token_ids = token_ids[: self.max_seq_len]
                    if len(token_ids) == 0:
                        token_ids = [0]
                    input_ids = torch.tensor([token_ids], dtype=torch.long)
                    captured.clear()
                    self.model(input_ids)
                    # captured[0]: (1, d_model)
                    results.append(captured[0])  # (1, d_model)
        finally:
            handle.remove()

        return torch.cat(results, dim=0)  # (N, d_model)

    # ------------------------------------------------------------------
    # Extraction methods
    # ------------------------------------------------------------------

    def extract_mean_diff(
        self,
        positive_texts: list[str],
        negative_texts: list[str],
    ) -> SteeringVector:
        """direction = normalize(mean(pos_hidden) - mean(neg_hidden))."""
        pos_hidden = self._get_hidden_states(positive_texts)  # (N, D)
        neg_hidden = self._get_hidden_states(negative_texts)  # (N, D)
        direction = pos_hidden.mean(dim=0) - neg_hidden.mean(dim=0)  # (D,)
        direction = F.normalize(direction, dim=0)
        return SteeringVector(
            direction=direction,
            layer_idx=self.layer_idx,
            label="steering",
            source="mean_diff",
        )

    def extract_pca(
        self,
        positive_texts: list[str],
        negative_texts: list[str],
    ) -> SteeringVector:
        """Compute (pos - neg) for each pair, stack into matrix, get first PC.

        Uses torch.pca_lowrank(X, q=1) -> (U, S, V); first PC = V[:, 0].
        Normalizes to unit norm.
        """
        pos_hidden = self._get_hidden_states(positive_texts)  # (N, D)
        neg_hidden = self._get_hidden_states(negative_texts)  # (N, D)
        diff_matrix = pos_hidden - neg_hidden  # (N, D)
        # Center the matrix
        diff_matrix = diff_matrix - diff_matrix.mean(dim=0, keepdim=True)
        _U, _S, V = torch.pca_lowrank(diff_matrix, q=1)
        direction = V[:, 0]  # (D,)
        direction = F.normalize(direction, dim=0)
        return SteeringVector(
            direction=direction,
            layer_idx=self.layer_idx,
            label="steering",
            source="pca",
        )

    def extract_multiple_layers(
        self,
        positive_texts: list[str],
        negative_texts: list[str],
        layer_indices: list[int],
    ) -> list[SteeringVector]:
        """Extract mean_diff steering vector for each layer in layer_indices."""
        vectors: list[SteeringVector] = []
        original_layer_idx = self.layer_idx
        for idx in layer_indices:
            self.layer_idx = idx
            sv = self.extract_mean_diff(positive_texts, negative_texts)
            sv.layer_idx = idx
            vectors.append(sv)
        self.layer_idx = original_layer_idx
        return vectors


# ---------------------------------------------------------------------------
# ActivationSteerer
# ---------------------------------------------------------------------------

class ActivationSteerer:
    """Apply steering vectors to model activations during forward pass.

    Registers forward hooks on specified layers that add:
        hidden_state += alpha * steering_vector.direction

    Args:
        model: AureliusTransformer instance.
        steering_vectors: List of SteeringVector to apply.
        alpha: Steering strength (e.g. 10.0 to 20.0).

    Usage:
        with ActivationSteerer(model, [sv], alpha=10.0):
            output = model(input_ids)
    """

    def __init__(
        self,
        model: torch.nn.Module,
        steering_vectors: list[SteeringVector],
        alpha: float = 10.0,
    ) -> None:
        self.model = model
        self.steering_vectors = steering_vectors
        self.alpha = alpha
        self._handles: list = []

    def __enter__(self) -> "ActivationSteerer":
        """Register all hooks."""
        self._handles.clear()
        for sv in self.steering_vectors:
            direction = sv.direction  # (D,)
            alpha = self.alpha

            def make_hook(dir_: torch.Tensor, a: float):
                def hook(module, inputs, output):
                    if isinstance(output, tuple):
                        hidden = output[0]
                        rest = output[1:]
                        hidden = hidden + a * dir_.to(hidden.device)
                        return (hidden,) + rest
                    else:
                        return output + a * dir_.to(output.device)
                return hook

            handle = self.model.layers[sv.layer_idx].register_forward_hook(
                make_hook(direction, alpha)
            )
            self._handles.append(handle)
        return self

    def __exit__(self, *args) -> None:
        """Remove all hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def steer(
        self,
        steering_vectors: list[SteeringVector],
        alpha: float,
    ) -> "ActivationSteerer":
        """Return self as context manager with updated vectors and alpha."""
        self.steering_vectors = steering_vectors
        self.alpha = alpha
        return self


# ---------------------------------------------------------------------------
# compute_steering_effect
# ---------------------------------------------------------------------------

def compute_steering_effect(
    model: torch.nn.Module,
    tokenizer_encode: Callable[[str], list[int]],
    tokenizer_decode: Callable[[list[int]], str],
    prompt: str,
    steering_vector: SteeringVector,
    alphas: list[float],
    max_new_tokens: int = 20,
) -> dict:
    """Generate text with different steering strengths.

    Returns {alpha: generated_text} for each alpha in alphas.
    alpha=0 is baseline (no steering).
    """
    results: dict = {}

    token_ids = tokenizer_encode(prompt)
    if len(token_ids) == 0:
        token_ids = [0]
    input_ids = torch.tensor([token_ids], dtype=torch.long)

    for alpha in alphas:
        if alpha == 0.0:
            # Baseline: no steering
            model.eval()
            with torch.no_grad():
                output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)
        else:
            steerer = ActivationSteerer(model, [steering_vector], alpha=alpha)
            model.eval()
            with steerer:
                with torch.no_grad():
                    output_ids = model.generate(input_ids, max_new_tokens=max_new_tokens)

        # Decode only newly generated tokens
        generated_ids = output_ids[0, len(token_ids):].tolist()
        results[alpha] = tokenizer_decode(generated_ids)

    return results


# ---------------------------------------------------------------------------
# ContrastiveActivationDataset
# ---------------------------------------------------------------------------

class ContrastiveActivationDataset:
    """Simple container for positive/negative text pairs for steering vector extraction.

    Provides some default pair sets via from_preset().
    """

    HELPFUL_PAIRS = (
        ["I'd be happy to help with that!", "Sure, here's a clear explanation."],
        ["I cannot and will not help.", "I refuse to assist."]
    )

    HONEST_PAIRS = (
        ["I'm not certain, but I believe...", "The evidence suggests..."],
        ["Definitely! 100% true.", "Absolutely guaranteed!"]
    )

    def __init__(self, positive: list[str], negative: list[str]) -> None:
        assert len(positive) == len(negative), (
            f"positive and negative must have equal length, "
            f"got {len(positive)} vs {len(negative)}"
        )
        self.positive = positive
        self.negative = negative

    def __len__(self) -> int:
        return len(self.positive)

    @classmethod
    def from_preset(cls, name: str) -> "ContrastiveActivationDataset":
        """Create a dataset from a named preset.

        Args:
            name: 'helpful' | 'honest'

        Returns:
            ContrastiveActivationDataset with the preset pairs.
        """
        if name == "helpful":
            positive, negative = cls.HELPFUL_PAIRS
        elif name == "honest":
            positive, negative = cls.HONEST_PAIRS
        else:
            raise ValueError(f"Unknown preset: {name!r}. Choose from 'helpful', 'honest'.")
        return cls(positive=list(positive), negative=list(negative))
