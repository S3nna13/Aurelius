"""Hyena operator — sub-quadratic long convolution alternative to attention.

Reference: Poli et al. 2023 — "Hyena Hierarchy: Towards Larger Convolutional Language Models".

Key idea: Replace O(N²) attention with implicit long convolutions parameterized by a
small MLP, computed via FFT for O(N log N) complexity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HyenaFilter(nn.Module):
    """Generates the Hyena convolution kernel from positional encodings.

    A small MLP maps position indices to kernel values.
    This creates a learnable, data-independent convolution kernel.

    Args:
        d_model: model dimension
        order: Hyena order (number of projections, default 2)
        kernel_len: maximum sequence length (= max_seq_len)
        d_filter: filter MLP hidden dim (default 64)
    """

    def __init__(
        self,
        d_model: int,
        order: int = 2,
        kernel_len: int = 512,
        d_filter: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.order = order
        self.kernel_len = kernel_len

        # MLP: position → kernel values
        # Input: 1D position index (normalized to [0, 1])
        # Output: (order * d_model) kernel values per position
        self.filter_mlp = nn.Sequential(
            nn.Linear(1, d_filter),
            nn.SiLU(),
            nn.Linear(d_filter, d_filter),
            nn.SiLU(),
            nn.Linear(d_filter, order * d_model),
        )

        # Positional encodings: [0, 1, ..., kernel_len-1] / kernel_len
        positions = torch.linspace(0, 1, kernel_len).unsqueeze(1)  # (L, 1)
        self.register_buffer("positions", positions)

    def forward(self, seq_len: int) -> torch.Tensor:
        """Generate filter for given sequence length.

        Args:
            seq_len: desired sequence length (<= kernel_len)

        Returns:
            Tensor of shape (order, d_model, seq_len)
        """
        pos = self.positions[:seq_len]  # (seq_len, 1)
        h = self.filter_mlp(pos)        # (seq_len, order * d_model)
        # Reshape to (order, d_model, seq_len)
        h = h.view(seq_len, self.order, self.d_model)  # (seq_len, order, d_model)
        h = h.permute(1, 2, 0)                          # (order, d_model, seq_len)
        return h


class HyenaOperator(nn.Module):
    """Hyena operator: O(N log N) alternative to attention.

    For order=2 (standard):
    1. Project input x to (order+1) streams: v, z1, z2 = in_proj(x)
    2. For each order i:
       - Generate kernel h_i from HyenaFilter
       - Compute: y = fft_conv(z_i, h_i) * y  (element-wise gating)
    3. Final linear projection back to d_model

    Args:
        d_model: model dimension
        order: Hyena order (default 2)
        kernel_len: max sequence length
        d_filter: filter network hidden dim
    """

    def __init__(
        self,
        d_model: int,
        order: int = 2,
        kernel_len: int = 512,
        d_filter: int = 64,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.order = order

        # Project input to (order+1) parallel streams
        self.in_proj = nn.Linear(d_model, (order + 1) * d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        self.filter = HyenaFilter(d_model, order, kernel_len, d_filter)

    def fft_conv(self, u: torch.Tensor, k: torch.Tensor) -> torch.Tensor:
        """Fast causal convolution via FFT.

        Args:
            u: (B, d_model, L) — input signal
            k: (d_model, L) — convolution kernel (causal)

        Returns:
            Tensor of shape (B, d_model, L)

        Algorithm:
        1. Pad both u and k to length 2L (circular → linear conv)
        2. FFT both: U = fft(u, n=2L), K = fft(k, n=2L)
        3. Multiply pointwise: Y = U * K
        4. IFFT: y = ifft(Y)
        5. Take first L elements (causal output)
        """
        seq_len = u.shape[-1]
        fft_len = 2 * seq_len

        # FFT along the sequence dimension
        U = torch.fft.rfft(u, n=fft_len, dim=-1)   # (B, d_model, fft_len//2+1)
        K = torch.fft.rfft(k, n=fft_len, dim=-1)   # (d_model, fft_len//2+1)

        # Broadcast and multiply
        Y = U * K.unsqueeze(0)                       # (B, d_model, fft_len//2+1)

        # IFFT and take causal part
        y = torch.fft.irfft(Y, n=fft_len, dim=-1)   # (B, d_model, fft_len)
        return y[..., :seq_len]                       # (B, d_model, L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)

        Returns:
            out: (B, L, d_model)
        """
        B, L, _ = x.shape

        # 1. Project to (order+1) streams
        proj = self.in_proj(x)                          # (B, L, (order+1)*d_model)

        # 2. Split into [v, z_1, ..., z_order]
        streams = proj.split(self.d_model, dim=-1)      # list of (B, L, d_model), len=order+1
        v = streams[0]                                  # (B, L, d_model) — gating signal
        zs = streams[1:]                                # order streams

        # 3. Transpose to (B, d_model, L) for conv
        v = v.transpose(1, 2)                           # (B, d_model, L)
        zs = [z.transpose(1, 2) for z in zs]           # list of (B, d_model, L)

        # 4. Generate all kernels: (order, d_model, L)
        kernels = self.filter(L)                        # (order, d_model, L)

        # 5. Recurrent gating: y = fft_conv(z_i, h_i) * y
        y = v
        for i in range(self.order):
            y = self.fft_conv(zs[i], kernels[i]) * y   # (B, d_model, L)

        # 6. Transpose back and project
        y = y.transpose(1, 2)                           # (B, L, d_model)
        out = self.out_proj(y)                          # (B, L, d_model)
        return out


class HyenaBlock(nn.Module):
    """Drop-in replacement for a TransformerBlock using Hyena instead of attention.

    Uses pre-norm (RMSNorm) + residual connections, matching the standard
    Aurelius transformer block interface.
    """

    def __init__(self, config) -> None:
        super().__init__()
        from .ffn import SwiGLUFFN
        from .rms_norm import RMSNorm

        self.norm1 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.hyena = HyenaOperator(config.d_model, kernel_len=config.max_seq_len)
        self.norm2 = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn = SwiGLUFFN(config)

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """Standard pre-norm residual. Ignores freqs_cis/mask kwargs for compatibility.

        Args:
            x: (B, L, d_model)

        Returns:
            out: (B, L, d_model)
        """
        x = x + self.hyena(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
