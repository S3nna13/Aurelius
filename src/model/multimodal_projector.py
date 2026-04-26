"""Multimodal Projector — maps external modality features into Aurelius token space.

Inspired by TriBEv2's modality-specific feature extractors that project into a
shared representation space, and LLaVA's approach of prepending visual tokens
to the input sequence.

Usage:
    cfg = ProjectorConfig(input_dim=768, output_dim=2048)
    proj = ModalityProjector(cfg)
    # x: (B, N, 768) → out: (B, N, 2048)

    mm = MultiModalProjector({"vision": cfg_v, "audio": cfg_a})
    seq = mm.build_multimodal_sequence(text_embeds, {"vision": img_feats})
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class ProjectorConfig:
    """Configuration for a single modality projector MLP."""

    input_dim: int  # source feature dim (e.g. 768 for CLIP)
    output_dim: int  # Aurelius d_model (e.g. 2048)
    hidden_dim: int | None = None  # if None, use (input_dim + output_dim) // 2
    n_layers: int = 2  # MLP depth (2 = one hidden layer)
    activation: str = "gelu"  # "gelu" | "relu" | "silu"
    dropout: float = 0.0
    layer_norm_input: bool = True  # normalize input features first


# ---------------------------------------------------------------------------
# ModalityProjector
# ---------------------------------------------------------------------------

_ACTIVATIONS: dict[str, type[nn.Module]] = {
    "gelu": nn.GELU,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
}


class ModalityProjector(nn.Module):
    """MLP that projects external modality features into token embedding space.

    Architecture (n_layers=2):
        [LayerNorm] → Linear(input_dim, hidden_dim) → activation → Linear(hidden_dim, output_dim)

    For n_layers=1: Linear(input_dim, output_dim) only.
    For n_layers=3: adds one more (hidden_dim, hidden_dim) → activation layer.
    """

    def __init__(self, cfg: ProjectorConfig) -> None:
        super().__init__()

        if cfg.activation not in _ACTIVATIONS:
            raise ValueError(
                f"Unknown activation '{cfg.activation}'. Choose from {list(_ACTIVATIONS.keys())}."
            )
        if cfg.n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {cfg.n_layers}.")

        self.hidden_dim: int = (
            cfg.hidden_dim if cfg.hidden_dim is not None else (cfg.input_dim + cfg.output_dim) // 2
        )

        act_cls = _ACTIVATIONS[cfg.activation]

        layers: list[nn.Module] = []

        # Optional input layer norm
        if cfg.layer_norm_input:
            layers.append(nn.LayerNorm(cfg.input_dim))

        if cfg.n_layers == 1:
            layers.append(nn.Linear(cfg.input_dim, cfg.output_dim))
        else:
            # First layer: input → hidden
            layers.append(nn.Linear(cfg.input_dim, self.hidden_dim))
            layers.append(act_cls())
            if cfg.dropout > 0.0:
                layers.append(nn.Dropout(cfg.dropout))

            # Middle layers: hidden → hidden  (only when n_layers > 2)
            for _ in range(cfg.n_layers - 2):
                layers.append(nn.Linear(self.hidden_dim, self.hidden_dim))
                layers.append(act_cls())
                if cfg.dropout > 0.0:
                    layers.append(nn.Dropout(cfg.dropout))

            # Final layer: hidden → output
            layers.append(nn.Linear(self.hidden_dim, cfg.output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        """Project features into token embedding space.

        Args:
            x: (B, N, input_dim) or (B, input_dim)

        Returns:
            Tensor with same leading dims and last dim = output_dim.
        """
        return self.net(x)


# ---------------------------------------------------------------------------
# MultiModalProjector
# ---------------------------------------------------------------------------


class MultiModalProjector(nn.Module):
    """Manages multiple modality-specific projectors.

    Each modality has its own ModalityProjector.
    Supports any string key: "vision", "audio", "depth", "structured", etc.
    """

    def __init__(self, modality_configs: dict[str, ProjectorConfig]) -> None:
        super().__init__()
        self.projectors = nn.ModuleDict(
            {name: ModalityProjector(cfg) for name, cfg in modality_configs.items()}
        )

    def project(self, modality: str, features: Tensor) -> Tensor:
        """Project a single modality's features.

        Args:
            modality: key registered in modality_configs.
            features: (B, N, input_dim) or (B, input_dim).

        Returns:
            Projected tensor with last dim = output_dim.
        """
        if modality not in self.projectors:
            raise KeyError(
                f"Unknown modality '{modality}'. Registered: {list(self.projectors.keys())}."
            )
        return self.projectors[modality](features)

    def project_all(self, features_dict: dict[str, Tensor]) -> dict[str, Tensor]:
        """Project all modalities in the dict.

        Args:
            features_dict: modality_name → (B, N, input_dim).

        Returns:
            Dict with same keys, values projected to output_dim.
        """
        return {name: self.project(name, feats) for name, feats in features_dict.items()}

    def build_multimodal_sequence(
        self,
        text_embeds: Tensor,
        modality_features: dict[str, Tensor],
        prepend_order: list[str] | None = None,
    ) -> Tensor:
        """Project all modalities and prepend them to text_embeds.

        Args:
            text_embeds: (B, S, d_model) — text token embeddings.
            modality_features: modality_name → (B, N, input_dim).
            prepend_order: order to prepend modalities. None = dict insertion order.

        Returns:
            (B, sum_N + S, d_model) — all tokens concatenated.
        """
        order = prepend_order if prepend_order is not None else list(modality_features.keys())
        projected = [self.project(name, modality_features[name]) for name in order]
        # Concatenate projected modality tokens then text tokens along sequence dim
        return torch.cat([*projected, text_embeds], dim=1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def build_projector(input_dim: int, output_dim: int, **kwargs) -> ModalityProjector:
    """Convenience factory: create a ModalityProjector from keyword args.

    Args:
        input_dim: source feature dimensionality.
        output_dim: target embedding dimensionality.
        **kwargs: passed to ProjectorConfig (hidden_dim, n_layers, activation, etc.)

    Returns:
        Configured ModalityProjector.
    """
    cfg = ProjectorConfig(input_dim=input_dim, output_dim=output_dim, **kwargs)
    return ModalityProjector(cfg)
