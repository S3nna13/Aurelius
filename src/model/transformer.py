"""Aurelius — Decoder-only transformer with GQA, SwiGLU, RoPE, and RMSNorm."""

from __future__ import annotations

import torch
import torch.nn as nn

from .attention import GroupedQueryAttention, precompute_rope_frequencies
from .config import AureliusConfig
from .ffn import SwiGLUFFN
from .rms_norm import RMSNorm


class TransformerBlock(nn.Module):
    """Single decoder layer: pre-norm attention + pre-norm FFN with residuals."""

    def __init__(self, config: AureliusConfig, layer_idx: int = 0) -> None:
        super().__init__()
        self.layer_idx = layer_idx

        # Determine if this layer uses RoPE
        nope = config.nope_every_n_layers
        apply_rope = True if nope == 0 else ((layer_idx + 1) % nope != 0)

        # Select attention class
        if config.use_diff_attn:
            from .attention import DifferentialAttention
            self.attn = DifferentialAttention(config, apply_rope=apply_rope)
        else:
            self.attn = GroupedQueryAttention(config, apply_rope=apply_rope)

        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), freqs_cis, mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class AureliusTransformer(nn.Module):
    """Aurelius 1.3B decoder-only transformer.

    Architecture highlights:
        - 24 transformer blocks
        - Grouped-Query Attention (16 Q heads, 8 KV heads)
        - SwiGLU FFN (d_ff = 5632)
        - RoPE positional encoding (theta = 500,000)
        - RMSNorm pre-normalization
        - Tied input/output embeddings
        - No bias in any linear layer
    """

    def __init__(self, config: AureliusConfig | None = None) -> None:
        super().__init__()
        self.config = config or AureliusConfig()

        # Token embedding
        self.embed = nn.Embedding(self.config.vocab_size, self.config.d_model)

        # Transformer blocks
        self.layers = nn.ModuleList(
            [TransformerBlock(self.config, layer_idx=i) for i in range(self.config.n_layers)]
        )

        # Final norm
        self.norm = RMSNorm(self.config.d_model, eps=self.config.rms_norm_eps)

        # Output head (tied with embedding)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        if self.config.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # Precompute RoPE frequencies (buffer — not a parameter, moves with .to())
        self.register_buffer(
            "freqs_cis",
            precompute_rope_frequencies(
                self.config.head_dim,
                self.config.max_seq_len,
                self.config.rope_theta,
            ),
            persistent=False,
        )

        # Initialize weights
        self._init_weights()

    def _init_weights(self) -> None:
        """Small normal init for linear layers, scaled by depth for residual paths."""
        std = 0.02
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)

    def forward(
        self,
        input_ids: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: (batch, seq_len) — token indices.
            mask: Optional attention mask broadcastable to (B, H, S, S).

        Returns:
            Logits of shape (batch, seq_len, vocab_size).
        """
        B, S = input_ids.shape
        assert S <= self.config.max_seq_len, (
            f"Sequence length {S} exceeds max_seq_len {self.config.max_seq_len}"
        )

        x = self.embed(input_ids)
        freqs_cis = self.freqs_cis[:S]

        for layer in self.layers:
            x = layer(x, freqs_cis, mask)

        x = self.norm(x)
        return self.lm_head(x)

    def count_parameters(self, *, count_embeddings: bool = True) -> dict[str, int]:
        """Count model parameters broken down by component.

        Args:
            count_embeddings: Whether to include embedding parameters.
                With tied embeddings, the embedding matrix is shared with lm_head,
                so we count it only once.

        Returns:
            Dictionary mapping component name to parameter count.
        """
        counts: dict[str, int] = {}

        # Embedding (counted once even when tied)
        if count_embeddings:
            counts["embedding"] = sum(p.numel() for p in self.embed.parameters())

        # Per-layer breakdown for first layer (all layers are identical)
        layer0 = self.layers[0]
        attn_params = sum(p.numel() for p in layer0.attn.parameters())
        ffn_params = sum(p.numel() for p in layer0.ffn.parameters())
        norm_params = sum(p.numel() for p in layer0.attn_norm.parameters()) + sum(
            p.numel() for p in layer0.ffn_norm.parameters()
        )
        per_layer = attn_params + ffn_params + norm_params

        counts["attention_per_layer"] = attn_params
        counts["ffn_per_layer"] = ffn_params
        counts["norm_per_layer"] = norm_params
        counts["per_layer_total"] = per_layer
        counts["all_layers"] = per_layer * self.config.n_layers

        # Final norm
        counts["final_norm"] = sum(p.numel() for p in self.norm.parameters())

        # LM head (0 if tied, since we already counted embedding)
        if self.config.tie_embeddings:
            counts["lm_head (tied)"] = 0
        else:
            counts["lm_head"] = sum(p.numel() for p in self.lm_head.parameters())

        # Total unique parameters
        counts["total"] = sum(p.numel() for p in self.parameters())

        return counts


def count_parameters(model: nn.Module) -> int:
    """Count total trainable parameters in any PyTorch model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    config = AureliusConfig()
    print(f"Aurelius Configuration:")
    print(f"  Layers:      {config.n_layers}")
    print(f"  Hidden dim:  {config.d_model}")
    print(f"  FFN dim:     {config.d_ff}")
    print(f"  Q heads:     {config.n_heads}")
    print(f"  KV heads:    {config.n_kv_heads}")
    print(f"  Head dim:    {config.head_dim}")
    print(f"  Vocab size:  {config.vocab_size:,}")
    print(f"  Max seq len: {config.max_seq_len:,}")
    print(f"  RoPE theta:  {config.rope_theta:,.0f}")
    print()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    model = AureliusTransformer(config).to(device)

    # Parameter breakdown
    param_counts = model.count_parameters()
    print(f"\nParameter Breakdown:")
    for name, count in param_counts.items():
        print(f"  {name:.<30} {count:>15,}")

    total = count_parameters(model)
    print(f"\nTotal trainable parameters: {total:,} ({total / 1e9:.3f}B)")

    # Verify forward pass
    print("\nRunning forward pass verification...")
    batch_size = 2
    seq_len = 128
    tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len), device=device)

    with torch.no_grad():
        logits = model(tokens)

    print(f"  Input shape:  {tokens.shape}")
    print(f"  Output shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, config.vocab_size), "Shape mismatch!"
    print("  Forward pass OK.")
