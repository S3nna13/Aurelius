"""FNet: Mixing Tokens with Fourier Transforms (Lee-Thorp et al., arXiv:2105.03824).

Replaces self-attention with a 2D Fourier Transform applied over the (sequence, hidden)
dimensions.  Token mixing costs O(T log T) vs O(T²) for attention, while retaining
~92 % of BERT accuracy according to the paper.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Core mixing primitive
# ---------------------------------------------------------------------------

class FourierMixingLayer(nn.Module):
    """Parameter-free token mixing via 2-D FFT.

    Applies ``torch.fft.fftn`` over the sequence and hidden dimensions, then
    discards the imaginary part, returning a real tensor of the same shape.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Mix tokens with a 2-D FFT.

        Args:
            x: Float tensor of shape ``(B, T, d_model)``.

        Returns:
            Real-valued tensor of the same shape ``(B, T, d_model)``.
        """
        # 2-D FFT over the last two axes: sequence (dim=-2) and hidden (dim=-1)
        x_fft = torch.fft.fftn(x, dim=(-2, -1))
        return x_fft.real


# ---------------------------------------------------------------------------
# FNet block
# ---------------------------------------------------------------------------

class FNetBlock(nn.Module):
    """Full FNet block: Fourier mixing + FFN, both in pre-norm style.

    Architecture:
        x = x + FourierMixing(mixing_norm(x))
        x = x + FFN(ffn_norm(x))

    where FFN is ``Linear → GELU → Linear → Dropout``.
    """

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0) -> None:
        super().__init__()

        self.mixing_norm = nn.LayerNorm(d_model)
        self.fourier = FourierMixingLayer()

        self.ffn_norm = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, d_ff)
        self.ffn_out = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Float tensor of shape ``(B, T, d_model)``.

        Returns:
            Float tensor of shape ``(B, T, d_model)``.
        """
        # --- Fourier mixing sub-layer ---
        x = x + self.fourier(self.mixing_norm(x))

        # --- FFN sub-layer ---
        residual = x
        x = self.ffn_norm(x)
        x = self.ffn_in(x)
        x = F.gelu(x)
        x = self.ffn_out(x)
        x = self.dropout(x)
        x = residual + x

        return x


# ---------------------------------------------------------------------------
# Full FNet model
# ---------------------------------------------------------------------------

class FNetModel(nn.Module):
    """Stack of FNetBlocks with token and positional embeddings.

    Args:
        vocab_size:     Vocabulary size for the token embedding.
        d_model:        Hidden / model dimension.
        d_ff:           Feed-forward intermediate dimension.
        n_layers:       Number of FNetBlock layers.
        max_seq_len:    Maximum sequence length (for learned positional embeddings).
        dropout:        Dropout probability applied inside each FNetBlock FFN.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int,
        d_ff: int,
        n_layers: int,
        max_seq_len: int = 512,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)

        self.blocks = nn.ModuleList(
            [FNetBlock(d_model, d_ff, dropout) for _ in range(n_layers)]
        )
        self.n_layers = n_layers

    def forward(
        self,
        input_ids: torch.Tensor,
        return_hidden_states: bool = False,
    ) -> torch.Tensor | list[torch.Tensor]:
        """Forward pass.

        Args:
            input_ids:            Long tensor of shape ``(B, T)``.
            return_hidden_states: When ``True``, return a list of hidden-state
                                  tensors (one per layer) instead of the final
                                  output tensor.

        Returns:
            ``(B, T, d_model)`` tensor, or a list of ``n_layers`` such tensors
            when ``return_hidden_states=True``.
        """
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)  # (1, T)

        x = self.token_emb(input_ids) + self.pos_emb(positions)

        hidden_states: list[torch.Tensor] = []
        for block in self.blocks:
            x = block(x)
            if return_hidden_states:
                hidden_states.append(x)

        if return_hidden_states:
            return hidden_states
        return x
