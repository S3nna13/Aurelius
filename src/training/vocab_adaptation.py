"""Vocabulary adaptation for domain-specific fine-tuning of Aurelius LLM.

Supports expanding model vocabulary with new tokens, initializing their
embeddings intelligently, and training with differentiated learning rates.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class VocabAdaptConfig:
    """Configuration for vocabulary adaptation."""
    new_tokens: list[str] = field(default_factory=list)
    init_method: str = "mean"          # "mean" | "random" | "closest"
    freeze_old_embeddings: bool = False
    scale_lr_new: float = 10.0         # LR multiplier for new token embeddings


# ---------------------------------------------------------------------------
# Similarity utilities
# ---------------------------------------------------------------------------

def compute_token_similarity(
    embed_weight: Tensor,
    query_embed: Tensor,
    top_k: int = 5,
) -> tuple[Tensor, Tensor]:
    """Find the top_k most similar existing embeddings to query_embed.

    Args:
        embed_weight: shape (V, D) — existing embedding matrix.
        query_embed:  shape (D,)  — query embedding vector.
        top_k:        number of nearest neighbours to return.

    Returns:
        (similarities, indices) both shape (top_k,).
    """
    # Normalise for cosine similarity
    norm_weight = F.normalize(embed_weight, dim=-1)          # (V, D)
    norm_query = F.normalize(query_embed.unsqueeze(0), dim=-1)  # (1, D)
    sims = (norm_weight @ norm_query.T).squeeze(-1)           # (V,)
    top_k = min(top_k, sims.shape[0])
    similarities, indices = torch.topk(sims, k=top_k)
    return similarities, indices


# ---------------------------------------------------------------------------
# Embedding initialisation
# ---------------------------------------------------------------------------

def initialize_new_token_embedding(
    embed_weight: Tensor,
    method: str = "mean",
    reference_indices: list[int] | None = None,
) -> Tensor:
    """Create an embedding vector for a new token.

    Args:
        embed_weight:      shape (V, D) — existing embedding matrix.
        method:            "mean" | "random" | "closest".
        reference_indices: token indices to average (used by "closest").

    Returns:
        Tensor of shape (D,).
    """
    if method == "mean":
        return embed_weight.mean(dim=0)

    elif method == "random":
        std = embed_weight.std().item()
        return torch.randn(embed_weight.shape[1], device=embed_weight.device) * std

    elif method == "closest":
        if reference_indices:
            idx = torch.tensor(reference_indices, device=embed_weight.device)
            return embed_weight[idx].mean(dim=0)
        # Fallback to mean
        return embed_weight.mean(dim=0)

    else:
        raise ValueError(f"Unknown init_method: {method!r}. Choose 'mean', 'random', or 'closest'.")


# ---------------------------------------------------------------------------
# Vocabulary expansion
# ---------------------------------------------------------------------------

def expand_vocabulary(model: nn.Module, n_new_tokens: int, method: str = "mean") -> None:
    """Expand model.embed and model.lm_head by n_new_tokens rows.

    New embedding rows are initialised according to *method*.
    Modifies the model in-place.

    Args:
        model:        AureliusTransformer (or any module with .embed and .lm_head).
        n_new_tokens: number of new vocabulary entries to add.
        method:       initialisation method — "mean" | "random" | "closest".
    """
    if n_new_tokens <= 0:
        return

    old_embed_weight: Tensor = model.embed.weight.data  # (V, D)
    V_old, D = old_embed_weight.shape

    # Build new rows for the embedding table
    new_rows = torch.stack(
        [initialize_new_token_embedding(old_embed_weight, method=method) for _ in range(n_new_tokens)],
        dim=0,
    )  # (n_new_tokens, D)

    new_embed_weight = torch.cat([old_embed_weight, new_rows], dim=0)  # (V+n, D)

    # Replace nn.Embedding
    new_embed = nn.Embedding(V_old + n_new_tokens, D)
    new_embed.weight = nn.Parameter(new_embed_weight)
    model.embed = new_embed

    # Replace nn.Linear (lm_head): shape (V, D) → (V+n, D)
    old_lm_weight: Tensor = model.lm_head.weight.data  # (V, D)
    new_lm_rows = torch.stack(
        [initialize_new_token_embedding(old_lm_weight, method=method) for _ in range(n_new_tokens)],
        dim=0,
    )
    new_lm_weight = torch.cat([old_lm_weight, new_lm_rows], dim=0)  # (V+n, D)

    bias = model.lm_head.bias
    new_lm_head = nn.Linear(D, V_old + n_new_tokens, bias=bias is not None)
    new_lm_head.weight = nn.Parameter(new_lm_weight)
    if bias is not None:
        new_bias = torch.cat([bias.data, torch.zeros(n_new_tokens, device=bias.device)], dim=0)
        new_lm_head.bias = nn.Parameter(new_bias)
    model.lm_head = new_lm_head

    # Update config if present
    if hasattr(model, "config") and hasattr(model.config, "vocab_size"):
        model.config.vocab_size = V_old + n_new_tokens


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------

class VocabAdaptationTrainer:
    """Fine-tuning trainer that adapts a pre-trained model to new vocabulary.

    Handles differentiated learning rates and optional freezing of old
    embedding rows so only new token representations are updated.
    """

    def __init__(
        self,
        model: nn.Module,
        config: VocabAdaptConfig,
        base_vocab_size: int,
    ) -> None:
        self.model = model
        self.config = config
        self.base_vocab_size = base_vocab_size
        self._n_new_tokens: int = model.embed.weight.shape[0] - base_vocab_size
        self._grad_hook_handle = None

    # ------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------

    def build_optimizer(self, base_lr: float = 1e-4) -> torch.optim.Optimizer:
        """Create AdamW with two param groups.

        New token embeddings use ``base_lr * scale_lr_new``; everything else
        uses ``base_lr``.
        """
        # Identify new token embedding rows by creating a view parameter
        # We keep the full embed.weight in group 1 but will override grad for old rows.
        new_embed_params = [self.model.embed.weight]  # full weight; hook clips old rows
        other_params = [
            p for name, p in self.model.named_parameters()
            if name != "embed.weight" and p.requires_grad
        ]

        param_groups = [
            {"params": new_embed_params, "lr": base_lr * self.config.scale_lr_new},
            {"params": other_params,     "lr": base_lr},
        ]
        return torch.optim.AdamW(param_groups, lr=base_lr)

    # ------------------------------------------------------------------
    # Training step
    # ------------------------------------------------------------------

    def train_step(
        self,
        input_ids: Tensor,
        labels: Tensor | None = None,
    ) -> dict:
        """Run one forward+backward step.

        Args:
            input_ids: (B, T) integer tensor of token IDs.
            labels:    (B, T) integer tensor; if None, shift input_ids by 1.

        Returns:
            {"loss": float, "n_new_tokens": int}
        """
        self.model.train()

        if labels is None:
            # Standard causal LM shift: predict next token
            labels = input_ids[:, 1:].contiguous()
            input_ids = input_ids[:, :-1].contiguous()

        output = self.model(input_ids)
        # Support plain tuple or object with .logits
        if isinstance(output, tuple):
            logits = output[1]
        else:
            logits = output.logits

        # logits: (B, T, V), labels: (B, T)
        B, T, V = logits.shape
        loss = F.cross_entropy(logits.reshape(B * T, V), labels.reshape(B * T))

        loss.backward()

        return {"loss": loss.item(), "n_new_tokens": self._n_new_tokens}

    # ------------------------------------------------------------------
    # Freezing
    # ------------------------------------------------------------------

    def freeze_pretrained(self) -> None:
        """Freeze all parameters except the new token embeddings.

        Practical approach: freeze all params with requires_grad_(False),
        then re-enable gradients for embed.weight and attach a hook to
        zero-out gradients for the old (base) rows after each backward pass.
        """
        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad_(False)

        # Unfreeze the full embedding weight so autograd runs
        self.model.embed.weight.requires_grad_(True)

        # Remove previous hook if any
        if self._grad_hook_handle is not None:
            self._grad_hook_handle.remove()

        base = self.base_vocab_size

        def _zero_old_rows(grad: Tensor) -> Tensor:
            """Zero gradient for old token rows, keeping only new ones."""
            modified = grad.clone()
            modified[:base] = 0.0
            return modified

        self._grad_hook_handle = self.model.embed.weight.register_hook(_zero_old_rows)


# ---------------------------------------------------------------------------
# Token usage analysis
# ---------------------------------------------------------------------------

def analyze_token_usage(
    input_ids: Tensor,
    vocab_size: int,
    top_k: int = 20,
) -> dict:
    """Analyse token frequency distribution in a batch of token IDs.

    Args:
        input_ids:  integer tensor of any shape.
        vocab_size: total vocabulary size.
        top_k:      number of most-frequent tokens to return.

    Returns:
        {
            "token_freq":  dict[int, int]  — token_id → count,
            "coverage":    float           — fraction of vocab appearing at least once,
            "top_tokens":  list[int]       — top_k most frequent token IDs,
        }
    """
    flat = input_ids.reshape(-1).tolist()
    counter = Counter(flat)
    token_freq: dict[int, int] = dict(counter)

    coverage = len(token_freq) / max(vocab_size, 1)

    top_tokens = [tok for tok, _ in counter.most_common(top_k)]

    return {
        "token_freq": token_freq,
        "coverage": coverage,
        "top_tokens": top_tokens,
    }
