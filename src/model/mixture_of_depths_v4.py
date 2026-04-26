"""
Mixture of Depths (MoD) — Raposo et al. 2024
Dynamic per-token compute allocation via learned routing.
Pure PyTorch, no external dependencies.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TokenRouter(nn.Module):
    """Scalar router for per-token routing decisions.

    Produces a boolean selection mask and raw logits.
    capacity_factor fraction of tokens are selected per batch item.
    """

    def __init__(self, d_model: int, capacity_factor: float = 0.5):
        super().__init__()
        self.capacity_factor = capacity_factor
        self.proj = nn.Linear(d_model, 1, bias=False)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, D)
        Returns:
            selected_mask: (B, T) bool — True for tokens that pass through
            router_logits: (B, T) float — raw scalar scores
        """
        B, T, D = x.shape
        k = max(1, int(T * self.capacity_factor))

        logits = self.proj(x).squeeze(-1)  # (B, T)

        # Top-k selection per batch item
        _, topk_indices = torch.topk(logits, k, dim=-1)  # (B, k)
        mask = torch.zeros(B, T, dtype=torch.bool, device=x.device)
        mask.scatter_(1, topk_indices, True)

        return mask, logits


class MoDBlock(nn.Module):
    """Transformer block with Mixture of Depths token routing.

    Selected tokens go through attention + FFN.
    Non-selected tokens pass through via residual connection unchanged.
    """

    def __init__(self, d_model: int, n_heads: int, capacity_factor: float = 0.5):
        super().__init__()
        self.router = TokenRouter(d_model, capacity_factor)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, T, D)
        Returns:
            output: (B, T, D) — same shape as input
            aux_loss: scalar tensor — router regularisation loss
        """
        B, T, D = x.shape
        mask, router_logits = self.router(x)  # (B, T), (B, T)

        # Auxiliary loss: prevent router collapse
        aux_loss = (router_logits**2).mean() * 0.01

        # Start with residual copy; we will overwrite selected positions
        output = x.clone()

        # Process each batch item independently (indices differ per item)
        for b in range(B):
            sel = mask[b]  # (T,) bool
            if sel.sum() == 0:
                continue  # all bypass — nothing to do

            x_sel = x[b][sel].unsqueeze(0)  # (1, k, D)

            # Self-attention on selected tokens
            normed = self.norm1(x_sel)
            attn_out, _ = self.attn(normed, normed, normed)
            x_sel = x_sel + attn_out

            # FFN
            x_sel = x_sel + self.ffn(self.norm2(x_sel))

            output[b][sel] = x_sel.squeeze(0)

        return output, aux_loss


class MoDTransformer(nn.Module):
    """Stack of MoD blocks with embedding and LM head."""

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        n_heads: int,
        vocab_size: int,
        capacity_factors: list | None = None,
    ):
        super().__init__()
        if capacity_factors is None:
            capacity_factors = [0.5] * n_layers
        assert len(capacity_factors) == n_layers, "capacity_factors length must equal n_layers"  # noqa: S101

        self.embedding = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList([MoDBlock(d_model, n_heads, cf) for cf in capacity_factors])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, input_ids: torch.Tensor):
        """
        Args:
            input_ids: (B, T) long
        Returns:
            logits: (B, T, V)
            total_aux_loss: scalar tensor
        """
        x = self.embedding(input_ids)  # (B, T, D)
        total_aux_loss = torch.zeros(1, device=x.device, dtype=x.dtype)

        for block in self.blocks:
            x, aux = block(x)
            total_aux_loss = total_aux_loss + aux

        x = self.norm(x)
        logits = self.head(x)  # (B, T, V)
        return logits, total_aux_loss.squeeze()


class MoDLoss(nn.Module):
    """Combined cross-entropy + router auxiliary loss."""

    def __init__(self, aux_weight: float = 0.01):
        super().__init__()
        self.aux_weight = aux_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor, aux_loss: torch.Tensor):
        """
        Args:
            logits:   (B, T, V)
            targets:  (B, T) long
            aux_loss: scalar
        Returns:
            total_loss, ce_loss, aux_loss_weighted
        """
        B, T, V = logits.shape
        ce_loss = F.cross_entropy(
            logits.reshape(B * T, V),
            targets.reshape(B * T),
            ignore_index=-100,
        )
        aux_loss_weighted = self.aux_weight * aux_loss
        total_loss = ce_loss + aux_loss_weighted
        return total_loss, ce_loss, aux_loss_weighted


class CapacityAnalyzer:
    """Accumulate and summarise routing statistics across batches."""

    def __init__(self):
        self._records: list = []

    def record(self, mask: torch.Tensor):
        """Accumulate selected fraction. mask: (B, T) bool."""
        frac = mask.float().mean().item()
        self._records.append(frac)

    def compute_stats(self) -> dict:
        if not self._records:
            return {
                "mean_capacity": 0.0,
                "std_capacity": 0.0,
                "min_capacity": 0.0,
                "max_capacity": 0.0,
            }
        t = torch.tensor(self._records, dtype=torch.float32)
        return {
            "mean_capacity": t.mean().item(),
            "std_capacity": t.std(correction=0).item() if len(t) > 1 else 0.0,
            "min_capacity": t.min().item(),
            "max_capacity": t.max().item(),
        }

    def reset(self):
        self._records.clear()
