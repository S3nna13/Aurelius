"""AQLM: Additive Quantization for Language Models (Egiazarian et al. 2024).

Compresses weight matrices to sub-2-bit precision by representing each
column-group as a sum of M codewords drawn from M independent codebooks.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


@dataclass
class AQLMConfig:
    """Configuration for an AQLM-quantized linear layer."""

    in_features: int            # weight matrix: [out_features, in_features]
    out_features: int
    n_codebooks: int = 2        # M: number of additive codebooks
    codebook_size: int = 256    # K: codewords per codebook
    group_size: int = 8         # quantize groups of `group_size` input features together


class AQLMCodebook(nn.Module):
    """Single AQLM codebook: a learnable lookup table of codewords."""

    def __init__(self, codebook_size: int, group_size: int) -> None:
        super().__init__()
        self.codebook_size = codebook_size
        self.group_size = group_size
        # Store codewords as a parameter so gradients flow through them.
        self._weight = nn.Parameter(torch.randn(codebook_size, group_size) * 0.02)

    @property
    def weight(self) -> Tensor:
        """Codeword table: [codebook_size, group_size]."""
        return self._weight


class AQLMLinear(nn.Module):
    """Linear layer whose weights are stored in AQLM additive-codebook format.

    During a forward pass the weights are reconstructed (dequantized) from
    integer codes and codebooks, then used with ``F.linear``.  Gradients flow
    only through the codebook parameters.

    Args:
        config: :class:`AQLMConfig` describing the layer dimensions and
            quantization hyper-parameters.
    """

    def __init__(self, config: AQLMConfig) -> None:
        super().__init__()

        if config.in_features % config.group_size != 0:
            raise ValueError(
                f"in_features ({config.in_features}) must be divisible by "
                f"group_size ({config.group_size})."
            )

        self.config = config
        self.in_features = config.in_features
        self.out_features = config.out_features
        self.n_codebooks = config.n_codebooks
        self.codebook_size = config.codebook_size
        self.group_size = config.group_size
        self.n_groups = config.in_features // config.group_size

        # M independent codebooks, each with K codewords of dimension group_size.
        self.codebooks = nn.ModuleList(
            [
                AQLMCodebook(config.codebook_size, config.group_size)
                for _ in range(config.n_codebooks)
            ]
        )

        # Integer codes: which codeword to pick from each codebook.
        # Shape: [out_features, n_groups, n_codebooks]
        self.register_buffer(
            "codes",
            torch.zeros(
                config.out_features,
                self.n_groups,
                config.n_codebooks,
                dtype=torch.int16,
            ),
        )

        # Per-output-feature scale for improved numerical range.
        self.scale = nn.Parameter(torch.ones(config.out_features, 1))

    # ------------------------------------------------------------------
    # Quantization (analysis pass)
    # ------------------------------------------------------------------

    def quantize(self, weight: Tensor) -> None:
        """Quantize a full-precision weight into AQLM codes (in-place).

        For every output neuron and every group of ``group_size`` input
        features a greedy nearest-neighbour search is performed: iterating
        over codebooks in order, the best codeword is selected from the
        current residual and subtracted before moving to the next codebook.

        Args:
            weight: Float tensor of shape ``[out_features, in_features]``.
        """
        assert weight.shape == (self.out_features, self.in_features), (
            f"Expected weight shape ({self.out_features}, {self.in_features}), "
            f"got {tuple(weight.shape)}."
        )

        # Reshape weight to [out_features, n_groups, group_size].
        w = weight.detach().float().reshape(self.out_features, self.n_groups, self.group_size)
        new_codes = torch.zeros(
            self.out_features, self.n_groups, self.n_codebooks, dtype=torch.int16
        )

        for m, cb in enumerate(self.codebooks):
            # codewords: [K, group_size]
            codewords = cb.weight.detach().float()  # [K, group_size]

            # Compute residual norm between each group and each codeword.
            # w: [out, n_groups, group_size] → unsqueeze → [out, n_groups, 1, group_size]
            # codewords: [K, group_size] → [1, 1, K, group_size]
            diff = w.unsqueeze(2) - codewords.unsqueeze(0).unsqueeze(0)
            # diff: [out, n_groups, K, group_size]
            dists = (diff ** 2).sum(dim=-1)  # [out, n_groups, K]
            idx = dists.argmin(dim=-1)       # [out, n_groups]
            new_codes[:, :, m] = idx.to(torch.int16)

            # Subtract the chosen codewords from the residual.
            chosen = codewords[idx]          # [out, n_groups, group_size]
            w = w - chosen

        self.codes.copy_(new_codes)

    # ------------------------------------------------------------------
    # Reconstruction (synthesis pass)
    # ------------------------------------------------------------------

    def dequantize(self) -> Tensor:
        """Reconstruct the full-precision weight from codes and codebooks.

        Returns:
            Float tensor of shape ``[out_features, in_features]``.
        """
        # Accumulate contributions from each codebook.
        # Start with zeros: [out_features, n_groups, group_size]
        reconstructed = torch.zeros(
            self.out_features,
            self.n_groups,
            self.group_size,
            device=self.codes.device,
            dtype=self.scale.dtype,
        )

        for m, cb in enumerate(self.codebooks):
            idx = self.codes[:, :, m].long()  # [out_features, n_groups]
            codewords = cb.weight             # [K, group_size]
            # Gather selected codewords: [out_features, n_groups, group_size]
            selected = codewords[idx]
            reconstructed = reconstructed + selected

        # Reshape to [out_features, in_features] and apply per-row scale.
        weight = reconstructed.reshape(self.out_features, self.in_features)
        weight = weight * self.scale          # broadcast: [out_features, 1]
        return weight

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """Dequantize weight and apply as a linear layer.

        Args:
            x: Input tensor of shape ``[B, in_features]`` or
               ``[B, T, in_features]``.

        Returns:
            Output tensor of shape ``[B, out_features]`` or
            ``[B, T, out_features]``.
        """
        w = self.dequantize()  # [out_features, in_features]
        return F.linear(x, w)

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    def compression_ratio(self) -> float:
        """Compression ratio relative to an fp32 baseline.

        bits_per_weight = (n_codebooks * log2(codebook_size)) / group_size
        ratio = 32 / bits_per_weight

        Higher ratio means more aggressive compression.  More codebooks
        increase the bit-width (lower ratio) but improve reconstruction
        quality.

        Returns:
            Compression ratio (> 1.0 means we use fewer bits than fp32).
        """
        bits_per_weight = (
            self.n_codebooks * math.log2(self.codebook_size)
        ) / self.group_size
        return 32.0 / bits_per_weight
