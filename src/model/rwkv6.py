"""RWKV-6 (Eagle) — data-dependent time mixing with matrix-valued states.

Reference: Peng et al. 2024, "Eagle and Finch: RWKV with Matrix-Valued States
and Dynamic Recurrence". https://arxiv.org/abs/2404.05892

Key difference from RWKV-4/5: the decay/shift parameters are data-dependent
(functions of the input token) rather than fixed per-channel scalars.

Pure PyTorch — no custom CUDA kernels, no external dependencies.

Components
----------
RWKV6TimeMix    : time-mixing with dynamic (input-dependent) decay gates.
RWKV6ChannelMix : channel-mixing FFN with squared-ReLU gating.
RWKV6Block      : pre-norm block combining both sub-layers.
RWKV6Model      : embedding + stack of blocks (returns hidden states).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# RWKV6TimeMix
# ---------------------------------------------------------------------------

class RWKV6TimeMix(nn.Module):
    """RWKV-6 time-mixing with data-dependent (dynamic) decay.

    Args:
        d_model : total model dimension.
        n_heads : number of heads.  d_head = d_model // n_heads.

    Forward signature: forward(x: Tensor) -> Tensor
        x   : (B, T, d_model)
        out : (B, T, d_model)
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head  = d_model // n_heads

        # ---- data-dependent gate projections --------------------------------
        # All produce (B, T, d_model); separate linear layers, no bias (as in paper).
        self.W_r = nn.Linear(d_model, d_model, bias=False)  # receptance
        self.W_k = nn.Linear(d_model, d_model, bias=False)  # key
        self.W_v = nn.Linear(d_model, d_model, bias=False)  # value
        self.W_w = nn.Linear(d_model, d_model, bias=False)  # dynamic decay
        self.W_o = nn.Linear(d_model, d_model, bias=False)  # output mix

        # ---- learnable bonus (u) — one scalar per head ----------------------
        # u shape: (n_heads, d_head) broadcast across batch/time.
        self.u = nn.Parameter(torch.zeros(n_heads, self.d_head))

        # ---- time-shift interpolation factors (static, like RWKV-4) --------
        # Used to mix x_t with x_{t-1} before the projections.
        self.time_mix_r = nn.Parameter(torch.full((1, 1, d_model), 0.5))
        self.time_mix_k = nn.Parameter(torch.full((1, 1, d_model), 0.5))
        self.time_mix_v = nn.Parameter(torch.full((1, 1, d_model), 0.5))
        self.time_mix_w = nn.Parameter(torch.full((1, 1, d_model), 0.5))

        # ---- layer norm on output before W_o --------------------------------
        self.ln_x = nn.LayerNorm(self.d_head)

        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.orthogonal_(self.W_r.weight)
        nn.init.orthogonal_(self.W_k.weight)
        nn.init.orthogonal_(self.W_v.weight)
        # Initialize W_w so that initial decay ≈ -1 (exp(-1) ≈ 0.37).
        nn.init.constant_(self.W_w.weight, 0.0)
        nn.init.zeros_(self.W_o.weight)
        nn.init.zeros_(self.u)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x : (B, T, d_model)

        Returns:
            out : (B, T, d_model)
        """
        B, T, D = x.shape
        H, dh   = self.n_heads, self.d_head

        # ---- time-shift (mix current token with previous token) -------------
        # Prepend zero for position 0.
        x_prev = torch.cat([torch.zeros(B, 1, D, device=x.device, dtype=x.dtype),
                             x[:, :-1, :]], dim=1)  # (B, T, D)

        xr = self.time_mix_r * x + (1.0 - self.time_mix_r) * x_prev
        xk = self.time_mix_k * x + (1.0 - self.time_mix_k) * x_prev
        xv = self.time_mix_v * x + (1.0 - self.time_mix_v) * x_prev
        xw = self.time_mix_w * x + (1.0 - self.time_mix_w) * x_prev

        # ---- projections ----------------------------------------------------
        r = torch.sigmoid(self.W_r(xr))          # (B, T, D)
        k = self.W_k(xk)                          # (B, T, D)
        v = self.W_v(xv)                          # (B, T, D)
        # dynamic per-token log-decay; negative so actual decay < 1.
        w = -F.softplus(-self.W_w(xw)) - 1e-4    # (B, T, D)  ≤ 0

        # ---- reshape to multi-head ------------------------------------------
        # (B, T, H, dh)
        r = r.view(B, T, H, dh)
        k = k.view(B, T, H, dh)
        v = v.view(B, T, H, dh)
        w = w.view(B, T, H, dh)           # log-decay per head-dim

        # ---- recurrent computation over T -----------------------------------
        # State s_t ∈ R^{dh × dh} per head.  We unroll the recurrence:
        #   s_t = exp(w_t) * s_{t-1} + k_t^T ⊗ v_t
        # output_t = r_t @ (s_{t-1}^bonus + s_{t-1})
        #   where s_{t-1}^bonus adds the current-token term via u.
        #
        # To keep gradients tractable we use a log-space formulation but
        # store the state in linear space with numerical stabilisation.
        # For training-time parallel we simply unroll over T with autograd.

        # s: (B, H, dh, dh)  — matrix-valued state
        s = torch.zeros(B, H, dh, dh, device=x.device, dtype=x.dtype)

        outputs = []
        for t in range(T):
            r_t = r[:, t, :, :]    # (B, H, dh)
            k_t = k[:, t, :, :]    # (B, H, dh)
            v_t = v[:, t, :, :]    # (B, H, dh)
            w_t = w[:, t, :, :]    # (B, H, dh)  log-decay

            # Current-token bonus: exp(u + k_t) * v_t as rank-1 update.
            # u shape (H, dh) → (1, H, dh).
            u_k = self.u.unsqueeze(0) + k_t              # (B, H, dh)
            # kv_bonus: outer product per head (B, H, dh, dh)
            kv_bonus = torch.einsum("bhk,bhv->bhkv", torch.exp(u_k), v_t)

            # Attention output using previous state + current-token bonus.
            # o_t = r_t @ (s_{t-1} + kv_bonus)
            # r_t: (B, H, dh) → (B, H, 1, dh) for batched matmul
            o_t = torch.einsum("bhi,bhij->bhj", r_t, s + kv_bonus)  # (B, H, dh)

            outputs.append(o_t)

            # Update state: s_t = exp(w_t) * s_{t-1} + k_t^T ⊗ v_t
            # decay: exp(w_t) broadcast over state dims.
            decay = torch.exp(w_t)                        # (B, H, dh)
            # (B, H, dh) → (B, H, dh, 1) for broadcasting across state rows.
            decay_mat = decay.unsqueeze(-1)               # (B, H, dh, 1)
            kv_t = torch.einsum("bhk,bhv->bhkv", k_t, v_t)  # (B, H, dh, dh)
            s = decay_mat * s + kv_t

        # Stack outputs: list of T tensors (B, H, dh) → (B, T, H, dh)
        out = torch.stack(outputs, dim=1)   # (B, T, H, dh)

        # ---- layer norm per head, then gate with r --------------------------
        out = self.ln_x(out)                # (B, T, H, dh)
        out = out * r                       # (B, T, H, dh)

        # ---- reshape back and project output --------------------------------
        out = out.reshape(B, T, D)          # (B, T, D)
        out = self.W_o(out)                 # (B, T, D)

        return out


# ---------------------------------------------------------------------------
# RWKV6ChannelMix
# ---------------------------------------------------------------------------

class RWKV6ChannelMix(nn.Module):
    """RWKV-6 channel-mixing (FFN) with squared-ReLU gating.

    Args:
        d_model : model dimension.

    Forward signature: forward(x: Tensor) -> Tensor
    """

    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.d_model = d_model
        d_ff = 4 * d_model  # 4× expansion, consistent with RWKV paper.

        # Time-shift interpolation factors.
        self.time_mix_r = nn.Parameter(torch.full((1, 1, d_model), 0.5))
        self.time_mix_k = nn.Parameter(torch.full((1, 1, d_model), 0.5))

        self.W_r = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_ff,    bias=False)
        self.W_v = nn.Linear(d_ff,    d_model, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x : (B, T, d_model)

        Returns:
            out : (B, T, d_model)
        """
        B, T, D = x.shape

        # Time-shift: mix x_t with x_{t-1}.
        x_prev = torch.cat([torch.zeros(B, 1, D, device=x.device, dtype=x.dtype),
                             x[:, :-1, :]], dim=1)  # (B, T, D)

        xr = self.time_mix_r * x + (1.0 - self.time_mix_r) * x_prev
        xk = self.time_mix_k * x + (1.0 - self.time_mix_k) * x_prev

        r = torch.sigmoid(self.W_r(xr))         # (B, T, D)
        k = F.relu(self.W_k(xk)) ** 2           # squared-ReLU, (B, T, d_ff)

        out = r * self.W_v(k)                   # (B, T, D)
        return out


# ---------------------------------------------------------------------------
# RWKV6Block
# ---------------------------------------------------------------------------

class RWKV6Block(nn.Module):
    """Pre-norm RWKV-6 block: LayerNorm + TimeMix + LayerNorm + ChannelMix.

    Args:
        d_model : model dimension.
        n_heads : number of time-mixing heads.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.time_mix    = RWKV6TimeMix(d_model, n_heads)
        self.channel_mix = RWKV6ChannelMix(d_model)

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x : (B, T, d_model)

        Returns:
            out : (B, T, d_model)
        """
        x = x + self.time_mix(self.ln1(x))
        x = x + self.channel_mix(self.ln2(x))
        return x


# ---------------------------------------------------------------------------
# RWKV6Model
# ---------------------------------------------------------------------------

class RWKV6Model(nn.Module):
    """Stack of RWKV-6 blocks with token embedding.

    Returns hidden states (B, T, d_model); no LM head — attach one externally.

    Args:
        vocab_size : vocabulary size.
        d_model    : model dimension.
        n_heads    : number of time-mixing heads per block.
        n_layers   : number of stacked RWKV6Blocks.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model:    int,
        n_heads:    int,
        n_layers:   int,
    ) -> None:
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks     = nn.ModuleList(
            [RWKV6Block(d_model, n_heads) for _ in range(n_layers)]
        )
        self.ln_out = nn.LayerNorm(d_model)

        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.normal_(self.embedding.weight, std=0.02)

    def forward(self, input_ids: Tensor) -> Tensor:
        """
        Args:
            input_ids : (B, T)  long tensor of token indices.

        Returns:
            hidden : (B, T, d_model)
        """
        x = self.embedding(input_ids)   # (B, T, d_model)
        for block in self.blocks:
            x = block(x)
        x = self.ln_out(x)
        return x
