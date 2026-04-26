"""Linear Recurrent Unit (LRU) — stable complex-diagonal linear recurrence.

Reference: Orvieto et al., 2023, "Resurrecting Recurrent Neural Networks for
Long Sequences". https://arxiv.org/abs/2303.06349

The LRU replaces the traditional recurrent matrix with a complex-diagonal
transition A, parameterized so that |eigenvalues| < 1 is guaranteed without
any projection or clipping:

    h_t = diag(A) * h_{t-1} + B_tilde * x_t
    y_t = Re(C * h_t) + D * x_t

Parameterization:
    nu   = softplus(nu_log)          → positive
    theta = exp(theta_log)           → positive phase
    mag  = exp(-exp(nu))             → in (0, 1), stable by construction
    A    = mag * (cos(theta) + i*sin(theta))   [complex diagonal]
    gamma = sqrt(1 - mag^2)          → normalization factor
    B_tilde = gamma * (B_re + i*B_im)  → normalized input projection

Implementation uses real-valued arithmetic with explicit Re/Im separation
(no complex64 dtype dependency for broadest hardware compatibility).

Architecture:
    Parameters (all real):
        nu_log    : [d_state]           log-magnitude parameter
        theta_log : [d_state]           log-phase parameter
        B_re      : [d_state, d_model]  real part of input projection
        B_im      : [d_state, d_model]  imaginary part of input projection
        C_re      : [d_model, d_state]  real part of output projection
        C_im      : [d_model, d_state]  imaginary part of output projection
        D         : [d_model]           skip/direct connection

Input : (B, T, d_model)
Output: (B, T, d_model)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class LRUConfig:
    """Configuration for LRULayer.

    Args:
        d_model:   Token / channel dimension.
        d_state:   Size of complex recurrent state.  The actual real-valued
                   state has dimension 2 * d_state (Re and Im parts).
        r_min:     Minimum eigenvalue magnitude for |A|.
        r_max:     Maximum eigenvalue magnitude for |A|.  Must be < 1 for
                   stability; defaults to 0.9 (leaving headroom).
        max_phase: Maximum phase angle in radians (default 2π).
    """

    d_model: int = 2048
    d_state: int = 256
    r_min: float = 0.0
    r_max: float = 0.9
    max_phase: float = 6.283185307179586  # 2 * pi


# ---------------------------------------------------------------------------
# LRU Layer
# ---------------------------------------------------------------------------


class LRULayer(nn.Module):
    """Linear Recurrent Unit — stable complex-diagonal recurrence.

    Implements the LRU from Orvieto et al. (2023) using real-valued arithmetic.
    The recurrence is:

        h_t = diag(A) ⊙ h_{t-1} + gamma ⊙ (B_re @ x_t + i * B_im @ x_t)
        y_t = C_re @ h_t.re + C_im @ h_t.im + D * x_t

    where A is parameterized so that |A| < 1 is guaranteed at all times.

    Args:
        config: LRUConfig instance.
    """

    def __init__(self, config: LRUConfig) -> None:
        super().__init__()
        self.config = config
        d_model = config.d_model
        d_state = config.d_state
        r_min = config.r_min
        r_max = config.r_max
        max_phase = config.max_phase

        # ------------------------------------------------------------------ #
        # Eigenvalue magnitude parameterization                               #
        # nu_log initialized so that exp(-exp(softplus(nu_log))) spans        #
        # [r_min, r_max] uniformly across d_state channels.                  #
        # ------------------------------------------------------------------ #
        # mag = exp(-exp(nu))  where nu = softplus(nu_log)
        # Inverting: nu_log s.t. softplus(nu_log) = -log(mag)
        #            nu_log = log(exp(-log(mag)) - 1)   (softplus inverse)
        # We want mag to span [r_min, r_max].
        # Avoid log(0) at r_min=0 by clamping to a small positive value.
        u = torch.linspace(0, 1, d_state)  # [0, 1]
        mag_init = r_min + (r_max - r_min) * u  # [r_min, r_max]
        mag_init = mag_init.clamp(min=1e-6, max=1.0 - 1e-6)
        nu_init = -torch.log(mag_init)  # = exp(nu) s.t. mag = exp(-exp(nu))
        # nu = softplus(nu_log)  → nu_log = log(exp(nu) - 1)
        nu_log_init = torch.log(torch.expm1(nu_init).clamp(min=1e-6))
        self.nu_log = nn.Parameter(nu_log_init)

        # ------------------------------------------------------------------ #
        # Phase parameterization                                              #
        # theta = exp(theta_log), initialized to span (0, max_phase).        #
        # ------------------------------------------------------------------ #
        theta_init = torch.linspace(math.log(1e-4), math.log(max_phase), d_state)
        self.theta_log = nn.Parameter(theta_init)

        # ------------------------------------------------------------------ #
        # Input projection B (complex): [d_state, d_model]                   #
        # ------------------------------------------------------------------ #
        self.B_re = nn.Parameter(torch.randn(d_state, d_model) / math.sqrt(d_model))
        self.B_im = nn.Parameter(torch.randn(d_state, d_model) / math.sqrt(d_model))

        # ------------------------------------------------------------------ #
        # Output projection C (complex): [d_model, d_state]                  #
        # ------------------------------------------------------------------ #
        self.C_re = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))
        self.C_im = nn.Parameter(torch.randn(d_model, d_state) / math.sqrt(d_state))

        # ------------------------------------------------------------------ #
        # Skip connection D: [d_model]                                       #
        # ------------------------------------------------------------------ #
        self.D = nn.Parameter(torch.ones(d_model))

    # ---------------------------------------------------------------------- #
    # Derived quantities                                                      #
    # ---------------------------------------------------------------------- #

    def get_A(self) -> tuple[Tensor, Tensor]:
        """Return the complex diagonal A as (A_re, A_im).

        Parameterization:
            nu    = softplus(nu_log)        → positive magnitude exponent
            theta = exp(theta_log)          → positive phase
            mag   = exp(-exp(nu))           → in (0, 1), stability guaranteed
            A_re  = mag * cos(theta)
            A_im  = mag * sin(theta)

        Returns:
            A_re: Tensor of shape [d_state]
            A_im: Tensor of shape [d_state]
        """
        nu = F.softplus(self.nu_log)  # [d_state], > 0
        theta = torch.exp(self.theta_log)  # [d_state], > 0
        mag = torch.exp(-torch.exp(nu))  # [d_state], in (0, 1)
        A_re = mag * torch.cos(theta)  # [d_state]
        A_im = mag * torch.sin(theta)  # [d_state]
        return A_re, A_im

    def get_gamma(self) -> Tensor:
        """Return the normalization factor gamma = sqrt(1 - |A|^2).

        By construction |A| = mag = exp(-exp(softplus(nu_log))) < 1,
        so 1 - mag^2 > 0 and gamma > 0 always.

        Returns:
            gamma: Tensor of shape [d_state], all values in (0, 1].
        """
        nu = F.softplus(self.nu_log)
        mag = torch.exp(-torch.exp(nu))  # [d_state]
        gamma = torch.sqrt(torch.clamp(1.0 - mag * mag, min=0.0))  # [d_state]
        return gamma

    # ---------------------------------------------------------------------- #
    # Forward pass                                                            #
    # ---------------------------------------------------------------------- #

    def forward_sequential(self, x: Tensor) -> Tensor:
        """Sequential (step-by-step) recurrence — correct but O(T) loop.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        B, T, _ = x.shape
        d_state = self.config.d_state

        # Derive stable A and normalization factor once per forward pass.
        A_re, A_im = self.get_A()  # [d_state]
        gamma = self.get_gamma()  # [d_state]

        # Normalized B: gamma * (B_re + i*B_im)
        gB_re = gamma.unsqueeze(1) * self.B_re  # [d_state, d_model]
        gB_im = gamma.unsqueeze(1) * self.B_im  # [d_state, d_model]

        # Initialize complex hidden state to zeros.
        h_re = x.new_zeros(B, d_state)  # [B, d_state]
        h_im = x.new_zeros(B, d_state)  # [B, d_state]

        outputs = []
        for t in range(T):
            x_t = x[:, t, :]  # [B, d_model]

            # Complex input projection: Bu = gB @ x_t
            Bu_re = x_t @ gB_re.t()  # [B, d_state]
            Bu_im = x_t @ gB_im.t()  # [B, d_state]

            # Complex state update: h_t = A * h_{t-1} + Bu_t
            # (a + ib)(c + id) = (ac - bd) + i(ad + bc)
            new_h_re = A_re * h_re - A_im * h_im + Bu_re  # [B, d_state]
            new_h_im = A_re * h_im + A_im * h_re + Bu_im  # [B, d_state]

            h_re = new_h_re
            h_im = new_h_im

            # Output projection: y_t = Re(C * h_t) + D * x_t
            # Re(C * h) = C_re @ h_re - C_im @ h_im
            y_t = (
                h_re @ self.C_re.t()  # [B, d_model]
                - h_im @ self.C_im.t()  # [B, d_model]
                + self.D * x_t  # [B, d_model]
            )

            outputs.append(y_t)

        # Stack along time: [B, T, d_model]
        return torch.stack(outputs, dim=1)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass for LRULayer.

        Args:
            x: Input tensor of shape (B, T, d_model).

        Returns:
            Output tensor of shape (B, T, d_model).
        """
        return self.forward_sequential(x)

    def state_size(self) -> int:
        """Return the size of the real-valued recurrent state.

        The complex state has d_state components, represented as 2*d_state
        real values (Re and Im parts interleaved conceptually).

        Returns:
            2 * d_state
        """
        return 2 * self.config.d_state
