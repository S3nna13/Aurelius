"""Attention Rollout attribution (Abnar & Zuidema, arXiv:2005.00928).

Propagates attention maps through all transformer layers, accounting for
residual connections, to produce a single rolled-up attention matrix that
captures how information flows from input tokens to any target position.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor

# ---------------------------------------------------------------------------
# Core rollout engine
# ---------------------------------------------------------------------------


class AttentionRollout:
    """Compute attention rollout from a list of per-layer attention maps.

    Args:
        discard_ratio: Fraction in [0, 1) of the lowest attention weights to
            zero out *per row* before rollout.  Helps suppress noisy low-weight
            connections.  0.0 disables discarding (default).
        head_fusion: Strategy for fusing multiple attention heads into a single
            (T, T) map.  One of ``"mean"`` (default), ``"min"``, ``"max"``.
    """

    def __init__(
        self,
        discard_ratio: float = 0.0,
        head_fusion: str = "mean",
    ) -> None:
        if not 0.0 <= discard_ratio < 1.0:
            raise ValueError(f"discard_ratio must be in [0, 1), got {discard_ratio}")
        if head_fusion not in {"mean", "min", "max"}:
            raise ValueError(f"head_fusion must be 'mean', 'min', or 'max', got {head_fusion!r}")

        self.discard_ratio = discard_ratio
        self.head_fusion = head_fusion

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _fuse_heads(self, attn: Tensor) -> Tensor:
        """Fuse head dimension → (..., T, T).

        Accepts (..., H, T, T) or (..., T, T).  When the input already lacks a
        head dimension it is returned unchanged.
        """
        if attn.dim() < 3:
            raise ValueError(f"Attention map must be at least 3-D, got shape {attn.shape}")

        # Detect whether a head dimension is present by checking if the last
        # two dimensions are equal (T×T) and there is an extra leading dim.
        # We treat the last two dims as (T, T) and everything before as batch+head.
        if attn.dim() == 2:
            # Already (T, T)
            return attn

        # (T, T) — no head dim
        if attn.dim() == 3:
            # Could be (H, T, T) *or* (B, T, T).  We assume (H, T, T) when the
            # caller supplies an unbatched map.  The public API documents that
            # (H, T, T) is the unbatched form with heads, while (T, T) is the
            # fully fused form.  For the purpose of compute() we treat the
            # leading dimension as *heads* only when the input is unbatched
            # (i.e. no batch dimension).  Since we can't distinguish H from B
            # without extra context we fuse unconditionally here — the caller
            # normalises the input before calling _fuse_heads.
            if self.head_fusion == "mean":
                return attn.mean(dim=0)
            elif self.head_fusion == "min":
                return attn.min(dim=0).values
            else:  # max
                return attn.max(dim=0).values

        # (B, H, T, T) — fuse over dim=1
        if self.head_fusion == "mean":
            return attn.mean(dim=1)
        elif self.head_fusion == "min":
            return attn.min(dim=1).values
        else:  # max
            return attn.max(dim=1).values

    def _discard_low_weights(self, attn: Tensor) -> Tensor:
        """Zero out the bottom ``discard_ratio`` fraction of each attention row.

        Args:
            attn: (..., T, T) attention matrix (rows = query positions).

        Returns:
            Masked attention matrix of the same shape.
        """
        if self.discard_ratio == 0.0:
            return attn

        flat = attn.reshape(-1, attn.shape[-1])  # (*, T)
        # Compute per-row quantile threshold
        threshold = torch.quantile(flat, self.discard_ratio, dim=-1, keepdim=True)
        mask = flat >= threshold
        flat = flat * mask.float()
        return flat.reshape(attn.shape)

    @staticmethod
    def _add_residual_and_normalise(attn: Tensor) -> Tensor:
        """Apply residual shortcut (0.5 * A + 0.5 * I) then row-normalise.

        Args:
            attn: (..., T, T) attention matrix.

        Returns:
            Row-normalised (..., T, T) matrix.
        """
        T = attn.shape[-1]
        eye = torch.eye(T, dtype=attn.dtype, device=attn.device)
        # Broadcast identity to match leading batch dims
        a_res = 0.5 * attn + 0.5 * eye
        row_sums = a_res.sum(dim=-1, keepdim=True)
        return a_res / row_sums

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute(self, attention_maps: list[Tensor]) -> Tensor:
        """Compute attention rollout from a list of per-layer attention maps.

        Args:
            attention_maps: List of L tensors.  Each tensor may be shaped:

                * ``(T, T)``       — single layer, pre-fused, unbatched
                * ``(H, T, T)``    — single layer, multi-head, unbatched
                * ``(B, H, T, T)`` — single layer, multi-head, batched

                All maps must have the same T (sequence length) and the same
                batch size (if batched).

        Returns:
            Rollout matrix:

            * ``(T, T)``    if the input maps were unbatched
            * ``(B, T, T)`` if the input maps were batched
        """
        if not attention_maps:
            raise ValueError("attention_maps must be non-empty")

        # Determine whether we are in batched mode by inspecting the first map.
        first = attention_maps[0]
        batched = first.dim() == 4  # (B, H, T, T)

        rollout: Tensor | None = None

        for attn in attention_maps:
            if batched:
                # attn: (B, H, T, T)
                B, H, T, _ = attn.shape
                # Fuse heads → (B, T, T)
                fused = self._fuse_heads(attn)  # uses dim=1 branch
            elif attn.dim() == 3:
                # (H, T, T) — unbatched with heads
                fused = self._fuse_heads(attn)  # uses dim=0 branch
                T = fused.shape[-1]
            elif attn.dim() == 2:
                # (T, T) — already fused
                fused = attn
                T = fused.shape[-1]
            else:
                raise ValueError(f"Unexpected attention map shape {attn.shape}")

            # Apply discard ratio
            fused = self._discard_low_weights(fused)

            # Add residual + normalise
            a_res = self._add_residual_and_normalise(fused)

            # Accumulate rollout
            if rollout is None:
                T = a_res.shape[-1]
                if batched:
                    B = a_res.shape[0]
                    rollout = (
                        torch.eye(T, dtype=a_res.dtype, device=a_res.device)
                        .unsqueeze(0)
                        .expand(B, T, T)
                        .clone()
                    )
                else:
                    rollout = torch.eye(T, dtype=a_res.dtype, device=a_res.device)

            # Matrix multiply: rollout = rollout @ a_res
            rollout = torch.bmm(rollout, a_res) if batched else rollout @ a_res

        return rollout  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Higher-level attributor
# ---------------------------------------------------------------------------


class RolloutAttributor:
    """Attribute importance scores to input tokens from a target position.

    Args:
        rollout: An :class:`AttentionRollout` instance used for computation.
    """

    def __init__(self, rollout: AttentionRollout) -> None:
        self.rollout = rollout

    def attribute(
        self,
        attention_maps: list[Tensor],
        target_pos: int = 0,
    ) -> Tensor:
        """Return per-token importance scores as seen from ``target_pos``.

        Args:
            attention_maps: Same format as :meth:`AttentionRollout.compute`.
            target_pos: The query position whose row in the rollout matrix we
                extract.  For a CLS-token encoder use ``target_pos=0``; for a
                causal LM use the last generated token position.

        Returns:
            * ``(T,)`` if inputs were unbatched
            * ``(B, T)`` if inputs were batched

            Values are normalised to sum to 1 along the T dimension.
        """
        rollout_mat = self.rollout.compute(attention_maps)
        # rollout_mat: (T, T) or (B, T, T)

        if rollout_mat.dim() == 2:
            # Unbatched → (T,)
            scores = rollout_mat[target_pos]  # (T,)
        else:
            # Batched → (B, T)
            scores = rollout_mat[:, target_pos, :]  # (B, T)

        # Normalise
        scores = scores / scores.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        return scores


# ---------------------------------------------------------------------------
# Forward-hook utility
# ---------------------------------------------------------------------------


class AttentionRolloutHook:
    """Collect attention maps from a model via forward hooks.

    The hook assumes that the attention module returns *either*:

    * A single ``Tensor`` (the attention weight matrix), or
    * A ``tuple`` whose **first element** is the attention weight tensor.

    Args:
        model: The PyTorch module to instrument.
        attention_module_class: If given, hooks are installed on every
            sub-module that is an instance of this class.  If ``None``,
            hooks are installed on every sub-module whose name ends with
            ``"attn"`` (case-sensitive).
    """

    def __init__(
        self,
        model: nn.Module,
        attention_module_class: type | None = None,
    ) -> None:
        self.model = model
        self.attention_module_class = attention_module_class
        self._hooks: list = []
        self._maps: list[Tensor] = []

    # ------------------------------------------------------------------
    # Hook callback
    # ------------------------------------------------------------------

    def _hook_fn(self, module: nn.Module, inputs: tuple, output) -> None:  # noqa: ANN001
        """Capture the attention weight tensor from the module output."""
        if isinstance(output, tuple):
            attn_weights = output[0]
        elif isinstance(output, Tensor):
            attn_weights = output
        else:
            return  # Unknown output type — skip silently

        if isinstance(attn_weights, Tensor):
            self._maps.append(attn_weights.detach())

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def register(self) -> None:
        """Install forward hooks on the target attention modules."""
        self._maps.clear()
        self._hooks.clear()

        for name, module in self.model.named_modules():
            if self.attention_module_class is not None:
                if isinstance(module, self.attention_module_class):
                    handle = module.register_forward_hook(self._hook_fn)
                    self._hooks.append(handle)
            else:
                # Fall back: match modules whose name segment ends with "attn"
                leaf_name = name.split(".")[-1]
                if leaf_name == "attn":
                    handle = module.register_forward_hook(self._hook_fn)
                    self._hooks.append(handle)

    def remove(self) -> None:
        """Remove all registered hooks."""
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    def get_maps(self) -> list[Tensor]:
        """Return the list of attention maps collected since the last :meth:`register` call."""
        return list(self._maps)

    def clear_maps(self) -> None:
        """Discard accumulated attention maps without removing hooks."""
        self._maps.clear()
