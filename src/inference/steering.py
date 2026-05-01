"""Activation addition / steering vectors for inference-time behavior control.

Extracts steering vectors from contrastive pairs and adds them to hidden
states at a specific layer during generation to shift model behavior.

Reference: Zou et al. 2023 "Representation Engineering" (arXiv:2310.01405)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class SteeringConfig:
    layer_idx: int = 15  # which layer to apply steering
    alpha: float = 10.0  # steering intensity (multiplier)
    normalize: bool = True  # normalize steering vector to unit norm


@dataclass
class SteeringVector:
    """A learned steering direction in activation space."""

    direction: torch.Tensor  # (D,) unit-norm vector
    layer_idx: int
    name: str = ""  # e.g. "helpfulness", "honesty"

    @property
    def norm(self) -> float:
        return self.direction.norm().item()


def extract_hidden_states_at_layer(
    model: nn.Module,
    input_ids: torch.Tensor,  # (B, S)
    layer_idx: int,
    pool: str = "mean",  # "mean", "last", "first"
) -> torch.Tensor:
    """Extract pooled hidden states at a specific layer.

    Hooks into model.layers[layer_idx] to capture output.
    Returns (B, D) pooled hidden states.
    """
    captured: list[torch.Tensor] = []

    def hook(module, input, output):
        # TransformerBlock returns (hidden_state, kv_cache)
        hs = output[0] if isinstance(output, tuple) else output
        captured.append(hs.detach().clone())

    handle = model.layers[layer_idx].register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(input_ids)
    finally:
        handle.remove()

    # captured[0] is (B, S, D)
    hs = captured[0]  # (B, S, D)

    if pool == "mean":
        return hs.mean(dim=1)  # (B, D)
    elif pool == "last":
        return hs[:, -1, :]  # (B, D)
    elif pool == "first":
        return hs[:, 0, :]  # (B, D)
    else:
        raise ValueError(f"Unknown pool mode: {pool!r}. Choose 'mean', 'last', or 'first'.")


def compute_steering_vector(
    model: nn.Module,
    positive_ids: torch.Tensor,  # (N, S) — positive examples (target behavior)
    negative_ids: torch.Tensor,  # (N, S) — negative examples (anti-target behavior)
    layer_idx: int,
    pool: str = "mean",
    normalize: bool = True,
) -> SteeringVector:
    """Compute steering vector as mean(positive_states) - mean(negative_states).

    Args:
        positive_ids: Token IDs for positive/target examples (e.g., helpful responses)
        negative_ids: Token IDs for negative/anti-target examples (e.g., unhelpful responses)
        layer_idx: Which layer to extract states from
        pool: How to pool over sequence dimension
        normalize: If True, L2-normalize the direction vector

    Returns:
        SteeringVector with unit-norm direction
    """
    pos_states = extract_hidden_states_at_layer(model, positive_ids, layer_idx, pool)  # (N, D)
    neg_states = extract_hidden_states_at_layer(model, negative_ids, layer_idx, pool)  # (N, D)

    direction = pos_states.mean(dim=0) - neg_states.mean(dim=0)  # (D,)

    if normalize:
        direction = direction / (direction.norm() + 1e-8)

    return SteeringVector(direction=direction, layer_idx=layer_idx)


class SteeringHook:
    """Context manager that adds steering vectors during forward passes.

    Usage:
        vector = compute_steering_vector(model, pos_ids, neg_ids, layer_idx=15)
        with SteeringHook(model, vector, alpha=10.0) as hook:
            output = model.generate(input_ids, max_new_tokens=50)
    """

    def __init__(
        self,
        model: nn.Module,
        vectors: SteeringVector | list[SteeringVector],
        alpha: float = 10.0,
    ) -> None:
        self.model = model
        self.vectors = [vectors] if isinstance(vectors, SteeringVector) else vectors
        self.alpha = alpha
        self._hooks: list = []

    def __enter__(self) -> SteeringHook:
        """Register forward hooks that add steering vectors."""
        for sv in self.vectors:
            layer = self.model.layers[sv.layer_idx]
            direction = sv.direction.to(next(self.model.parameters()).device)

            def make_hook(d):
                def hook(module, input, output):
                    # output is (hidden_state, kv_cache, aux_loss) for TransformerBlock
                    h, kv, aux = output
                    h = h + self.alpha * d.unsqueeze(0).unsqueeze(0)
                    return (h, kv, aux)

                return hook

            self._hooks.append(layer.register_forward_hook(make_hook(direction)))
        return self

    def __exit__(self, *args) -> None:
        for h in self._hooks:
            h.remove()
        self._hooks.clear()


def generate_with_steering(
    model: nn.Module,
    input_ids: torch.Tensor,  # (1, S)
    steering_vector: SteeringVector,
    max_new_tokens: int = 64,
    alpha: float = 10.0,
    temperature: float = 1.0,
    top_p: float = 0.9,
    eos_token_id: int | None = None,
) -> torch.Tensor:
    """Generate tokens with a steering vector applied at the target layer.

    Returns generated token ids (S_new,) — only the newly generated tokens
    (not including the prompt).
    """
    with SteeringHook(model, steering_vector, alpha=alpha):
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            eos_token_id=eos_token_id,
        )
    # output_ids is (1, prompt_len + generated_len); return only generated tokens
    prompt_len = input_ids.shape[1]
    return output_ids[0, prompt_len:]
