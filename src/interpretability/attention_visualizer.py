"""
src/interpretability/attention_visualizer.py

Attention visualization utilities for transformer interpretability.

Pure PyTorch + NumPy — no external dependencies beyond stdlib.
"""

from __future__ import annotations

import torch
from torch import Tensor


class AttentionVisualizer:
    """Utilities for computing and formatting attention maps for display."""

    def __init__(self, max_seq_len: int = 128) -> None:
        if not isinstance(max_seq_len, int):
            raise TypeError(f"max_seq_len must be int, got {type(max_seq_len).__name__}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")
        self.max_seq_len = max_seq_len

    # ------------------------------------------------------------------
    # compute_attention_map
    # ------------------------------------------------------------------

    def compute_attention_map(self, attn_weights: Tensor) -> Tensor:
        """Average attention weights across heads.

        Parameters
        ----------
        attn_weights : Tensor, shape (n_heads, seq_len, seq_len)
            Attention weight matrix.

        Returns
        -------
        Tensor of shape (seq_len, seq_len) — mean over heads.
        """
        self._validate_3d_square(attn_weights, "attn_weights")
        return attn_weights.mean(dim=0)  # (seq_len, seq_len)

    # ------------------------------------------------------------------
    # highlight_top_tokens
    # ------------------------------------------------------------------

    def highlight_top_tokens(
        self,
        attn_map: Tensor,
        token_ids: list[int],
        top_k: int = 3,
    ) -> dict[int, list[tuple[int, float]]]:
        """For each query position, return the top-k attended positions with scores.

        Parameters
        ----------
        attn_map : Tensor, shape (seq_len, seq_len)
            Averaged attention map.
        token_ids : list[int]
            Token IDs for each sequence position (used only for length validation).
        top_k : int, default 3
            Number of top attended positions to return per query.

        Returns
        -------
        dict[int, list[tuple[int, float]]]
            Mapping from query position -> list of (attended_position, score).
        """
        self._validate_2d_square(attn_map, "attn_map")
        seq_len = attn_map.shape[0]

        if not isinstance(token_ids, list):
            raise TypeError(f"token_ids must be list, got {type(token_ids).__name__}")
        if len(token_ids) != seq_len:
            raise ValueError(
                f"token_ids length ({len(token_ids)}) must match attn_map seq_len ({seq_len})"
            )
        if not isinstance(top_k, int):
            raise TypeError(f"top_k must be int, got {type(top_k).__name__}")
        if top_k <= 0:
            raise ValueError(f"top_k must be positive, got {top_k}")
        if top_k > seq_len:
            raise ValueError(f"top_k ({top_k}) cannot exceed seq_len ({seq_len})")

        result: dict[int, list[tuple[int, float]]] = {}
        for i in range(seq_len):
            row = attn_map[i]  # (seq_len,)
            # torch.topk returns (values, indices)
            values, indices = torch.topk(row, k=top_k)
            result[i] = [
                (int(indices[j]), float(values[j])) for j in range(top_k)
            ]
        return result

    # ------------------------------------------------------------------
    # rollout_attention
    # ------------------------------------------------------------------

    def rollout_attention(self, attn_maps: list[Tensor]) -> Tensor:
        """Multi-layer attention rollout via matrix multiplication across layers.

        Parameters
        ----------
        attn_maps : list[Tensor]
            List of L attention matrices, each of shape (seq_len, seq_len).

        Returns
        -------
        Tensor of shape (seq_len, seq_len) — joint attention rollout.
        """
        if not isinstance(attn_maps, list):
            raise TypeError(f"attn_maps must be list, got {type(attn_maps).__name__}")
        if len(attn_maps) == 0:
            raise ValueError("attn_maps must contain at least one attention matrix")

        first = attn_maps[0]
        self._validate_2d_square(first, "attn_maps[0]")
        seq_len = first.shape[0]

        for idx, mat in enumerate(attn_maps[1:], start=1):
            self._validate_2d_square(mat, f"attn_maps[{idx}]")
            if mat.shape != (seq_len, seq_len):
                raise ValueError(
                    f"All attention maps must have shape ({seq_len}, {seq_len}), "
                    f"but attn_maps[{idx}] has shape {tuple(mat.shape)}"
                )

        device = first.device
        dtype = first.dtype
        result = torch.eye(seq_len, dtype=dtype, device=device)
        for mat in attn_maps:
            result = mat @ result
        return result

    # ------------------------------------------------------------------
    # normalize_for_display
    # ------------------------------------------------------------------

    def normalize_for_display(self, attn_map: Tensor) -> Tensor:
        """Min-max normalize attention map to [0, 1].

        Parameters
        ----------
        attn_map : Tensor, shape (seq_len, seq_len)
            Attention map to normalize.

        Returns
        -------
        Tensor of shape (seq_len, seq_len) with values in [0, 1].
        """
        self._validate_2d_square(attn_map, "attn_map")

        min_val = attn_map.min()
        max_val = attn_map.max()
        diff = max_val - min_val
        if diff == 0:
            return torch.zeros_like(attn_map)
        return (attn_map - min_val) / diff

    # ------------------------------------------------------------------
    # Internal validators
    # ------------------------------------------------------------------

    def _validate_3d_square(self, tensor: Tensor, name: str) -> None:
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.dim() != 3:
            raise ValueError(f"{name} must be 3-D, got {tensor.dim()}-D")
        if tensor.shape[1] != tensor.shape[2]:
            raise ValueError(
                f"{name} must be square in last two dims, got shape {tuple(tensor.shape)}"
            )
        if not tensor.dtype.is_floating_point:
            raise TypeError(f"{name} must be a floating-point dtype, got {tensor.dtype}")
        seq_len = tensor.shape[1]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"{name} seq_len ({seq_len}) exceeds max_seq_len ({self.max_seq_len})"
            )

    def _validate_2d_square(self, tensor: Tensor, name: str) -> None:
        if not isinstance(tensor, Tensor):
            raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}")
        if tensor.dim() != 2:
            raise ValueError(f"{name} must be 2-D, got {tensor.dim()}-D")
        if tensor.shape[0] != tensor.shape[1]:
            raise ValueError(
                f"{name} must be square, got shape {tuple(tensor.shape)}"
            )
        if not tensor.dtype.is_floating_point:
            raise TypeError(f"{name} must be a floating-point dtype, got {tensor.dtype}")
        seq_len = tensor.shape[0]
        if seq_len > self.max_seq_len:
            raise ValueError(
                f"{name} seq_len ({seq_len}) exceeds max_seq_len ({self.max_seq_len})"
            )


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

ATTENTION_VISUALIZER_REGISTRY = {
    "default": AttentionVisualizer,
}
