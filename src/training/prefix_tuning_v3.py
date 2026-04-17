"""Prefix Tuning v3 — Li & Liang (2021) + Soft Prompting — Lester et al. (2021).

Parameter-efficient fine-tuning via learnable continuous prefix/prompt tokens
prepended to keys and values in each attention layer, or prepended to the
embedding sequence (prompt-tuning variant).

References:
    Li, X. L., & Liang, P. (2021). Prefix-Tuning: Optimizing Continuous
        Prompts for Generation. ACL 2021. https://arxiv.org/abs/2101.00190
    Lester, B., Al-Rfou, R., & Constant, N. (2021). The Power of Scale for
        Parameter-Efficient Prompt Tuning. EMNLP 2021.
        https://arxiv.org/abs/2104.08691
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# SoftPromptEmbedding
# ---------------------------------------------------------------------------

class SoftPromptEmbedding(nn.Module):
    """Learnable prompt tokens in embedding space (Lester et al. 2021).

    Args:
        n_tokens:       Number of soft prompt tokens.
        d_model:        Model embedding dimension.
        init_from_vocab: If True and vocab_size > 0, the caller must supply
                        ``init_weight`` (the full token-embedding matrix,
                        shape [vocab_size, d_model]) to ``forward`` so that
                        the parameter is initialised from sampled vocab rows.
                        This class stores random init; the caller can replace
                        ``prompt_embeddings`` after construction if needed.
        vocab_size:     Size of vocabulary (used only when init_from_vocab).
    """

    def __init__(
        self,
        n_tokens: int,
        d_model: int,
        init_from_vocab: bool = False,
        vocab_size: int = 0,
    ) -> None:
        super().__init__()
        self.n_tokens = n_tokens
        self.d_model = d_model

        self.prompt_embeddings = nn.Parameter(
            torch.randn(n_tokens, d_model) * 0.02
        )

        # Optional: record vocab-init metadata for caller use
        self._init_from_vocab = init_from_vocab
        self._vocab_size = vocab_size

    def init_from_vocab_embeddings(self, embedding_weight: Tensor) -> None:
        """Replace prompt_embeddings with rows sampled from an embedding table.

        Args:
            embedding_weight: Shape (vocab_size, d_model).
        """
        vocab_size = embedding_weight.size(0)
        indices = torch.randint(0, vocab_size, (self.n_tokens,))
        with torch.no_grad():
            self.prompt_embeddings.copy_(embedding_weight[indices].detach())

    def forward(self, batch_size: int) -> Tensor:
        """Return prompt embeddings expanded to batch.

        Returns:
            Tensor of shape (batch_size, n_tokens, d_model).
        """
        # (n_tokens, d_model) -> (1, n_tokens, d_model) -> (B, n_tokens, d_model)
        return self.prompt_embeddings.unsqueeze(0).expand(batch_size, -1, -1)


# ---------------------------------------------------------------------------
# PrefixEncoder
# ---------------------------------------------------------------------------

class PrefixEncoder(nn.Module):
    """MLP reparameterization for prefix parameters (more stable than raw params).

    Projects a small learnable matrix through a 2-layer MLP to produce
    per-layer, per-head prefix K and V tensors.

    Args:
        n_tokens:          Number of prefix tokens.
        d_model:           Total model dimension (must be divisible by n_heads).
        n_layers:          Number of transformer layers.
        n_heads:           Number of attention heads.
        prefix_hidden_dim: Hidden dimension of the reparameterisation MLP.
    """

    def __init__(
        self,
        n_tokens: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        prefix_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.n_tokens = n_tokens
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads

        # Latent prefix representation
        self.prefix_params = nn.Parameter(
            torch.randn(n_tokens, prefix_hidden_dim) * 0.02
        )

        # MLP: (prefix_hidden_dim) -> (prefix_hidden_dim) -> (n_layers * 2 * d_model)
        output_dim = n_layers * 2 * d_model
        self.mlp = nn.Sequential(
            nn.Linear(prefix_hidden_dim, prefix_hidden_dim),
            nn.Tanh(),
            nn.Linear(prefix_hidden_dim, output_dim),
        )

    def forward(self, batch_size: int) -> Tensor:
        """Compute prefix K/V tensors for all layers.

        Returns:
            Tensor of shape (n_layers, 2, batch_size, n_heads, n_tokens, head_dim).
        """
        # prefix_params: (n_tokens, prefix_hidden_dim)
        # mlp output:    (n_tokens, n_layers * 2 * d_model)
        x = self.mlp(self.prefix_params)  # (n_tokens, n_layers * 2 * d_model)

        # Reshape to (n_tokens, n_layers, 2, n_heads, head_dim)
        x = x.view(self.n_tokens, self.n_layers, 2, self.n_heads, self.head_dim)

        # Reorder to (n_layers, 2, n_heads, n_tokens, head_dim)
        x = x.permute(1, 2, 3, 0, 4)  # (n_layers, 2, n_heads, n_tokens, head_dim)

        # Insert batch dim at position 2: (n_layers, 2, 1, n_heads, n_tokens, head_dim)
        # then expand -> (n_layers, 2, batch_size, n_heads, n_tokens, head_dim)
        x = x.unsqueeze(2).expand(-1, -1, batch_size, -1, -1, -1)

        return x


# ---------------------------------------------------------------------------
# PrefixAttention
# ---------------------------------------------------------------------------

class PrefixAttention(nn.Module):
    """Multi-head attention that prepends learnable prefix tokens to K and V.

    Args:
        d_model:         Total model dimension.
        n_heads:         Number of attention heads.
        n_prefix_tokens: Number of prefix tokens to prepend.
    """

    def __init__(self, d_model: int, n_heads: int, n_prefix_tokens: int) -> None:
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
            )
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.n_prefix_tokens = n_prefix_tokens
        self.scale = math.sqrt(self.head_dim)

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

        # Learnable per-layer prefix K and V tokens
        self.prefix_k = nn.Parameter(torch.randn(n_prefix_tokens, d_model) * 0.02)
        self.prefix_v = nn.Parameter(torch.randn(n_prefix_tokens, d_model) * 0.02)

    def _split_heads(self, x: Tensor) -> Tensor:
        """(B, T, D) -> (B, n_heads, T, head_dim)."""
        B, T, D = x.shape
        return x.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        """(B, n_heads, T, head_dim) -> (B, T, D)."""
        B, H, T, hd = x.shape
        return x.transpose(1, 2).contiguous().view(B, T, H * hd)

    def forward(
        self,
        x: Tensor,
        prefix_k_override: Optional[Tensor] = None,
        prefix_v_override: Optional[Tensor] = None,
    ) -> Tensor:
        """Compute prefix-augmented attention.

        Args:
            x:                Shape (B, T, D).
            prefix_k_override: If provided, shape (B, n_heads, n_prefix_tokens, head_dim).
                               Overrides stored prefix_k.
            prefix_v_override: If provided, shape (B, n_heads, n_prefix_tokens, head_dim).
                               Overrides stored prefix_v.

        Returns:
            Tensor of shape (B, T, D).
        """
        B, T, D = x.shape

        Q = self._split_heads(self.q_proj(x))   # (B, H, T, hd)
        K = self._split_heads(self.k_proj(x))   # (B, H, T, hd)
        V = self._split_heads(self.v_proj(x))   # (B, H, T, hd)

        if self.n_prefix_tokens > 0:
            if prefix_k_override is not None:
                # (B, n_heads, n_prefix_tokens, head_dim)
                pk = prefix_k_override
            else:
                # Expand stored prefix: (n_prefix_tokens, D) -> project -> split
                pk = self._split_heads(
                    self.k_proj(self.prefix_k.unsqueeze(0).expand(B, -1, -1))
                )  # (B, H, n_prefix, hd)

            if prefix_v_override is not None:
                pv = prefix_v_override
            else:
                pv = self._split_heads(
                    self.v_proj(self.prefix_v.unsqueeze(0).expand(B, -1, -1))
                )  # (B, H, n_prefix, hd)

            # Prepend prefix to K and V
            K = torch.cat([pk, K], dim=2)  # (B, H, n_prefix + T, hd)
            V = torch.cat([pv, V], dim=2)  # (B, H, n_prefix + T, hd)

        # Scaled dot-product attention
        attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_out = torch.matmul(attn_weights, V)  # (B, H, T, hd)

        out = self._merge_heads(attn_out)          # (B, T, D)
        return self.out_proj(out)


# ---------------------------------------------------------------------------
# PrefixTuningModel
# ---------------------------------------------------------------------------

class PrefixTuningModel(nn.Module):
    """Wraps a frozen backbone with trainable prefix / soft-prompt parameters.

    The backbone is expected to expose:
        - ``backbone.embedding``: nn.Embedding with weight of shape (V, D)
        - ``backbone.lm_head``:   nn.Linear mapping (B, T, D) -> (B, T, V)

    For simplicity this implements the *prompt-tuning* variant: prefix
    embeddings are concatenated before the first backbone layer. The backbone
    sees a longer sequence ``[prefix | input]`` and the caller trims prefix
    tokens from the output logits.

    Args:
        backbone:               The frozen model (must have .embedding and .lm_head).
        n_prefix_tokens:        Number of prefix tokens.
        d_model:                Embedding dimension.
        n_layers:               Number of transformer layers (for PrefixEncoder).
        n_heads:                Number of attention heads (for PrefixEncoder).
        use_reparameterization: If True, use PrefixEncoder; otherwise use
                                SoftPromptEmbedding per layer.
    """

    def __init__(
        self,
        backbone: nn.Module,
        n_prefix_tokens: int,
        d_model: int,
        n_layers: int,
        n_heads: int,
        use_reparameterization: bool = True,
        prefix_hidden_dim: int = 512,
    ) -> None:
        super().__init__()
        self.backbone = backbone
        self.n_prefix_tokens = n_prefix_tokens
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.use_reparameterization = use_reparameterization

        # Freeze ALL backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False

        # Trainable prefix parameters
        if n_prefix_tokens > 0:
            if use_reparameterization:
                self.prefix_encoder: Optional[PrefixEncoder] = PrefixEncoder(
                    n_tokens=n_prefix_tokens,
                    d_model=d_model,
                    n_layers=n_layers,
                    n_heads=n_heads,
                    prefix_hidden_dim=prefix_hidden_dim,
                )
                self.soft_prompts: Optional[nn.ModuleList] = None
            else:
                self.prefix_encoder = None
                self.soft_prompts = nn.ModuleList(
                    [
                        SoftPromptEmbedding(n_prefix_tokens, d_model)
                        for _ in range(n_layers)
                    ]
                )
        else:
            self.prefix_encoder = None
            self.soft_prompts = None

        # Trainable embedding for prefix tokens (prompt-tuning variant)
        if n_prefix_tokens > 0:
            self.prefix_embed = nn.Parameter(
                torch.randn(n_prefix_tokens, d_model) * 0.02
            )
        else:
            self.prefix_embed = None

    def forward(self, input_ids: Tensor) -> Tensor:
        """Run backbone with prefix prepended to the embedding sequence.

        Args:
            input_ids: Shape (B, T) — integer token IDs.

        Returns:
            Logits of shape (B, T, V) corresponding only to the *input* tokens
            (prefix token positions are trimmed).
        """
        B, T = input_ids.shape

        # Get token embeddings from backbone
        token_emb = self.backbone.embedding(input_ids)  # (B, T, D)

        if self.n_prefix_tokens > 0:
            # Expand learnable prefix embeddings
            prefix_emb = self.prefix_embed.unsqueeze(0).expand(B, -1, -1)  # (B, P, D)
            # Prepend to sequence
            combined = torch.cat([prefix_emb, token_emb], dim=1)  # (B, P+T, D)
        else:
            combined = token_emb  # (B, T, D)

        # Run through backbone's lm_head (backbone is a simple embedding+linear)
        logits_all = self.backbone.lm_head(combined)  # (B, P+T, V)

        # Trim prefix positions
        if self.n_prefix_tokens > 0:
            logits = logits_all[:, self.n_prefix_tokens:, :]  # (B, T, V)
        else:
            logits = logits_all  # (B, T, V)

        return logits


# ---------------------------------------------------------------------------
# PrefixTuningTrainer
# ---------------------------------------------------------------------------

class PrefixTuningTrainer:
    """Trains only the prefix/prompt parameters of a PrefixTuningModel.

    Args:
        model:     A PrefixTuningModel instance.
        optimizer: PyTorch optimizer configured over the trainable params.
    """

    def __init__(self, model: PrefixTuningModel, optimizer: torch.optim.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

    def _count_params(self):
        n_trainable = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        n_frozen = sum(
            p.numel() for p in self.model.parameters() if not p.requires_grad
        )
        return n_trainable, n_frozen

    def train_step(self, input_ids: Tensor, labels: Tensor) -> dict:
        """Perform one forward + backward pass.

        Args:
            input_ids: Shape (B, T).
            labels:    Shape (B, T) — integer token IDs for next-token targets.
                       Typically ``labels = input_ids`` with shift handled here.

        Returns:
            dict with keys: ``loss`` (float), ``n_trainable_params`` (int),
            ``n_frozen_params`` (int).
        """
        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(input_ids)  # (B, T, V)

        # Shift for next-token prediction: predict token t+1 from position t
        # logits[:, :-1, :] predicts labels[:, 1:]
        shift_logits = logits[:, :-1, :].contiguous()          # (B, T-1, V)
        shift_labels = labels[:, 1:].contiguous()              # (B, T-1)

        B, Tm1, V = shift_logits.shape
        loss = F.cross_entropy(
            shift_logits.view(B * Tm1, V),
            shift_labels.view(B * Tm1),
        )

        loss.backward()
        self.optimizer.step()

        n_trainable, n_frozen = self._count_params()

        return {
            "loss": loss.item(),
            "n_trainable_params": n_trainable,
            "n_frozen_params": n_frozen,
        }
