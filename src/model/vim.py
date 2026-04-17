"""Vision Mamba (Vim) — bidirectional SSM for visual representation learning.

Reference: Zhu et al. 2024, "Vision Mamba: Efficient Visual Representation
Learning with Bidirectional State Space Model". arXiv:2401.13520.

Key idea: Patchify an image into a sequence of patch tokens, then process them
with a bidirectional Mamba SSM (forward scan + backward scan merged by
summation).  A learnable CLS token aggregates global context for classification.

Discretized state equations (Zero-Order Hold, linear-RNN form):
    A = -exp(A_log)                          [always negative → stable]
    A_bar_t = exp(Δ_t * A)                   [ZOH discrete A]
    B_bar_t = Δ_t * B_t                      [Euler approx for B]
    h_t = A_bar_t * h_{t-1} + B_bar_t * x_t [state update]
    y_t = C_t · h_t + D * x_t               [output + skip]
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# PatchEmbed
# ---------------------------------------------------------------------------


class PatchEmbed(nn.Module):
    """Convert a 2-D image into a sequence of patch embeddings.

    Uses a single Conv2d with kernel_size = stride = patch_size so every
    non-overlapping patch is projected to d_model dimensions in one pass.

    Args:
        img_size:   Expected input image size (H == W assumed for positional
                    embedding sizing, but the forward pass accepts any H/W).
        patch_size: Side length of each square patch (pixels).
        in_chans:   Number of input image channels.
        d_model:    Output embedding dimension.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        d_model: int = 192,
    ) -> None:
        super().__init__()
        self.patch_size = patch_size
        self.proj = nn.Conv2d(
            in_chans,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor.

        Returns:
            (B, N, d_model) where N = (H // patch_size) * (W // patch_size).
        """
        # (B, d_model, H//P, W//P)
        x = self.proj(x)
        # Flatten spatial dims and transpose: (B, N, d_model)
        B, D, Hg, Wg = x.shape
        x = x.reshape(B, D, Hg * Wg).transpose(1, 2)
        return x


# ---------------------------------------------------------------------------
# Core simplified SSM (single direction)
# ---------------------------------------------------------------------------


class _SSMCore(nn.Module):
    """Simplified Mamba-style SSM scan (one direction).

    Input-dependent Δ, B, C computed from x; A is a learned negative parameter.
    Sequential linear-RNN scan (ZOH discretization).

    Shape contract:
        input  : (B, T, d_inner)
        output : (B, T, d_inner)
    """

    def __init__(self, d_model: int, d_state: int, d_inner: int) -> None:
        super().__init__()
        self.d_inner = d_inner
        self.d_state = d_state

        dt_rank = max(1, math.ceil(d_model / 16))
        self.dt_rank = dt_rank

        # Combined projection: x → (Δ_raw, B_in, C_in)
        self.x_proj = nn.Linear(d_inner, dt_rank + 2 * d_state, bias=False)

        # Δ projection: dt_rank → d_inner
        self.dt_proj = nn.Linear(dt_rank, d_inner, bias=True)

        # Log-parameterised state matrix A: (d_inner, d_state)
        A_log = (
            torch.log(torch.arange(1, d_state + 1, dtype=torch.float32))
            .unsqueeze(0)
            .expand(d_inner, -1)
        )
        self.A_log = nn.Parameter(A_log.clone())
        self.A_log._no_weight_decay = True  # type: ignore[attr-defined]

        # D: skip connection scalar per channel
        self.D = nn.Parameter(torch.ones(d_inner))
        self.D._no_weight_decay = True  # type: ignore[attr-defined]

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, d_inner)

        Returns:
            (B, T, d_inner)
        """
        B, T, _ = x.shape

        xz = self.x_proj(x)  # (B, T, dt_rank + 2*d_state)
        delta_raw, B_in, C_in = xz.split([self.dt_rank, self.d_state, self.d_state], dim=-1)

        delta = F.softplus(self.dt_proj(delta_raw))  # (B, T, d_inner)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # A_bar: (B, T, d_inner, d_state)
        A_bar = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))

        # B_bar: (B, T, d_inner, d_state)
        B_bar = delta.unsqueeze(-1) * B_in.unsqueeze(2)

        # Sequential scan
        h = torch.zeros(B, self.d_inner, self.d_state, dtype=x.dtype, device=x.device)
        ys: list[Tensor] = []
        for t in range(T):
            h = A_bar[:, t] * h + B_bar[:, t] * x[:, t].unsqueeze(-1)
            y_t = (h * C_in[:, t].unsqueeze(1)).sum(dim=-1) + self.D * x[:, t]
            ys.append(y_t)

        return torch.stack(ys, dim=1)  # (B, T, d_inner)


# ---------------------------------------------------------------------------
# BidirectionalSSM
# ---------------------------------------------------------------------------


class BidirectionalSSM(nn.Module):
    """Mamba SSM applied in both forward and backward directions.

    Both directions share the same expanded inner dimension but have
    independent parameters.  Their outputs are summed and projected back to
    d_model.

    Args:
        d_model:  Input / output feature dimension.
        d_state:  SSM state dimension (N in the paper).
        expand:   Expansion factor; d_inner = expand * d_model.
    """

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2) -> None:
        super().__init__()
        d_inner = expand * d_model
        self.d_inner = d_inner

        # Project input up to d_inner for each direction
        self.in_proj_fwd = nn.Linear(d_model, d_inner, bias=False)
        self.in_proj_bwd = nn.Linear(d_model, d_inner, bias=False)

        # Independent SSM cores
        self.ssm_fwd = _SSMCore(d_model, d_state, d_inner)
        self.ssm_bwd = _SSMCore(d_model, d_state, d_inner)

        # Project merged result back to d_model
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model)
        """
        # Forward scan
        x_fwd = self.in_proj_fwd(x)            # (B, T, d_inner)
        y_fwd = self.ssm_fwd(x_fwd)            # (B, T, d_inner)

        # Backward scan: flip, scan, flip back
        x_bwd = self.in_proj_bwd(x.flip(1))    # (B, T, d_inner)  reversed
        y_bwd = self.ssm_bwd(x_bwd).flip(1)    # (B, T, d_inner)  unflipped

        # Merge by summation, then project
        return self.out_proj(y_fwd + y_bwd)    # (B, T, d_model)


# ---------------------------------------------------------------------------
# VimBlock
# ---------------------------------------------------------------------------


class VimBlock(nn.Module):
    """Vision Mamba block: pre-norm + BidirectionalSSM + residual.

    out = x + BidirectionalSSM(LayerNorm(x))

    Args:
        d_model:  Feature dimension.
        d_state:  SSM state dimension.
    """

    def __init__(self, d_model: int, d_state: int = 16) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = BidirectionalSSM(d_model, d_state=d_state)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, T, d_model)

        Returns:
            (B, T, d_model)
        """
        return x + self.ssm(self.norm(x))


# ---------------------------------------------------------------------------
# VisionMamba
# ---------------------------------------------------------------------------


class VisionMamba(nn.Module):
    """Full Vision Mamba image classification model.

    Pipeline:
        1. PatchEmbed: (B, C, H, W) → (B, N, d_model)
        2. Prepend learnable CLS token → (B, N+1, d_model)
        3. Add learnable position embeddings
        4. VimBlock stack
        5. Final LayerNorm
        6. CLS token → linear classifier head → (B, n_classes)

    Args:
        img_size:   Input image size (pixels, H == W).
        patch_size: Patch size (pixels).
        in_chans:   Image channels.
        d_model:    Hidden dimension.
        n_layers:   Number of VimBlock layers.
        n_classes:  Output classification classes.
        d_state:    SSM state dimension.
    """

    def __init__(
        self,
        img_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        d_model: int = 192,
        n_layers: int = 12,
        n_classes: int = 1000,
        d_state: int = 16,
    ) -> None:
        super().__init__()

        n_patches = (img_size // patch_size) ** 2

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, d_model)

        # Learnable CLS token: (1, 1, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        # Learnable position embeddings: CLS + N patches
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches + 1, d_model))

        self.blocks = nn.ModuleList(
            [VimBlock(d_model, d_state=d_state) for _ in range(n_layers)]
        )

        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)

        # Weight initialisation
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: (B, C, H, W) image tensor.

        Returns:
            (B, n_classes) logits.
        """
        B = x.shape[0]

        # Patch embedding: (B, N, d_model)
        x = self.patch_embed(x)

        # Prepend CLS token: (B, N+1, d_model)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add position embeddings
        x = x + self.pos_embed

        # VimBlock stack
        for block in self.blocks:
            x = block(x)

        # Final norm
        x = self.norm(x)

        # Classify using CLS token
        cls_out = x[:, 0]            # (B, d_model)
        return self.head(cls_out)    # (B, n_classes)
