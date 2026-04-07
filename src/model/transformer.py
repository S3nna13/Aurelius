"""Aurelius — Decoder-only transformer with GQA, SwiGLU, RoPE, and RMSNorm."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as ckpt

from .attention import GroupedQueryAttention, precompute_rope_frequencies, yarn_rope_frequencies
from .config import AureliusConfig
from .ffn import SwiGLUFFN
from .rms_norm import RMSNorm


class TransformerBlock(nn.Module):
    """Single decoder layer: pre-norm attention + pre-norm FFN with residuals."""

    def __init__(self, config: AureliusConfig) -> None:
        super().__init__()
        self.attn = GroupedQueryAttention(config)
        self.attn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.rms_norm_eps)
        self.ffn = SwiGLUFFN(config)

    def forward(
        self,
        x: torch.Tensor,
        freqs_cis: torch.Tensor,
        mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        attn_out, kv = self.attn(self.attn_norm(x), freqs_cis, mask, past_kv)
        x = x + attn_out
        x = x + self.ffn(self.ffn_norm(x))
        return x, kv



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
            [TransformerBlock(self.config) for _ in range(self.config.n_layers)]
        )

        # Final norm
        self.norm = RMSNorm(self.config.d_model, eps=self.config.rms_norm_eps)

        # Output head (tied with embedding)
        self.lm_head = nn.Linear(self.config.d_model, self.config.vocab_size, bias=False)
        if self.config.tie_embeddings:
            self.lm_head.weight = self.embed.weight

        # Precompute RoPE frequencies (buffer — not a parameter, moves with .to())
        if self.config.rope_scaling_type == "yarn":
            freqs = yarn_rope_frequencies(
                self.config.head_dim,
                self.config.max_seq_len,
                self.config.rope_theta,
                scale=self.config.rope_scaling_factor,
                original_max_seq_len=self.config.rope_original_max_seq_len,
            )
        else:
            freqs = precompute_rope_frequencies(
                self.config.head_dim,
                self.config.max_seq_len,
                self.config.rope_theta,
            )

        self.register_buffer("freqs_cis", freqs, persistent=False)

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
        labels: torch.Tensor | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor] | None] | None = None,
    ) -> tuple[torch.Tensor | None, torch.Tensor, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Args:
            input_ids: (batch, seq_len) — token indices.
            mask: Optional attention mask broadcastable to (B, H, S, S).
            labels: (batch, seq_len) — target token ids for computing cross-entropy loss.
            past_key_values: Per-layer KV cache from a previous forward pass.

        Returns:
            Tuple of (loss, logits, present_key_values):
                loss: Scalar cross-entropy loss if labels provided, else None.
                logits: (batch, seq_len, vocab_size).
                present_key_values: List of (k, v) tensors, one per layer.
        """
        B, S = input_ids.shape
        assert S <= self.config.max_seq_len, (
            f"Sequence length {S} exceeds max_seq_len {self.config.max_seq_len}"
        )

        # Compute position offset from KV cache
        past_len = (
            past_key_values[0][0].shape[1]
            if past_key_values is not None and past_key_values[0] is not None
            else 0
        )

        x = self.embed(input_ids)
        freqs_cis = self.freqs_cis[past_len : past_len + S]

        if self.config.use_gradient_checkpointing and past_key_values is not None:
            raise ValueError("Gradient checkpointing is incompatible with KV cache (past_key_values)")

        present_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        for i, layer in enumerate(self.layers):
            past_kv = past_key_values[i] if past_key_values is not None else None
            if self.config.use_gradient_checkpointing and self.training:
                # checkpoint needs all inputs as tensors; pass past_kv as None (no cache during training ckpt)
                def make_ckpt_fn(l):
                    def fn(x, freqs_cis, mask):
                        out, kv = l(x, freqs_cis, mask, None)
                        return out, kv[0], kv[1]
                    return fn
                x, k, v = ckpt(make_ckpt_fn(layer), x, freqs_cis, mask, use_reentrant=False)
                kv = (k, v)
            else:
                x, kv = layer(x, freqs_cis, mask, past_kv)
            present_key_values.append(kv)

        x = self.norm(x)
        logits = self.lm_head(x)

        loss = None
        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        return loss, logits, present_key_values

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int = 256,
        temperature: float = 1.0,
        top_p: float = 0.9,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation with top-p nucleus sampling and KV cache.

        Args:
            input_ids: (batch, prompt_len) — prompt token ids.
            max_new_tokens: Maximum tokens to generate.
            temperature: Sampling temperature (1.0 = no scaling).
            top_p: Nucleus sampling threshold.
            eos_token_id: Stop generation when this token is produced (all sequences).

        Returns:
            (batch, prompt_len + generated_len) token ids.
        """
        B, _ = input_ids.shape
        past_key_values = None
        cur_ids = input_ids

        for _ in range(max_new_tokens):
            _, logits, past_key_values = self(cur_ids, past_key_values=past_key_values)
            next_logits = logits[:, -1, :]  # (B, vocab)

            # Temperature scaling
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Top-p nucleus sampling (batch-correct)
            # Sort ASCENDING so cumsum is left-to-right from smallest probability
            sorted_logits, sorted_indices = torch.sort(next_logits, descending=False)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            # Remove tokens that push cumulative prob above (1 - top_p)
            sorted_mask = cumulative_probs <= (1.0 - top_p)
            # Always keep at least the top-1 token
            sorted_mask[..., -1:] = False
            # Scatter mask back to original vocabulary ordering
            mask = sorted_mask.scatter(1, sorted_indices, sorted_mask)
            next_logits = next_logits.masked_fill(mask, float("-inf"))

            next_token = torch.multinomial(next_logits.softmax(dim=-1), num_samples=1)  # (B, 1)

            cur_ids = next_token  # next step: only the new token (cache handles context)
            input_ids = torch.cat([input_ids, next_token], dim=1)

            if eos_token_id is not None and (next_token == eos_token_id).all():
                break

        return input_ids

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

        # Per-layer breakdown for first layer (all layers are identical).
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
        _, logits, _ = model(tokens)

    print(f"  Input shape:  {tokens.shape}")
    print(f"  Output shape: {logits.shape}")
    assert logits.shape == (batch_size, seq_len, config.vocab_size), "Shape mismatch!"
    print("  Forward pass OK.")
