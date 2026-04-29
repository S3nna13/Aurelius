"""Multi-Token Prediction (MTP) for Aurelius.

Implements parameter-shared MTP from GLM-5: N shared MTP layers for predicting
N future tokens. Improves training signal and enables speculative decoding.

References:
    - Nemotron 3 (NVIDIA, 2025): ~2.4% benchmark improvement, 97% 2-token acceptance
    - GLM-5 (Zhipu, 2026): Shared parameter MTP, 2.76 accept length vs 2.55
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class MTPLayer(nn.Module):
    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.norm = nn.RMSNorm(d_model)
        self.linear = nn.Linear(d_model * 2, d_model, bias=False)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, h: Tensor, h_prev: Tensor) -> tuple[Tensor, Tensor]:
        B, T, D = h.shape
        h_prev = h_prev[:, -T:, :]
        combined = torch.cat([h, h_prev], dim=-1)
        h_out = self.linear(combined)
        h_out = F.silu(h_out)
        h_out = self.norm(h_out)
        logits = self.head(h_out)
        return h_out, logits


class MTPModule(nn.Module):
    """Multi-Token Prediction module with shared parameters.

    Uses N shared MTP layers (same params, different positions in unrolled graph)
    to predict N future tokens. Provides auxiliary training loss and enables
    speculative decoding at inference.

    Args:
        d_model: Model hidden dimension
        vocab_size: Vocabulary size
        n_predict: Number of future tokens to predict (default: 2)
        share_params: Share parameters across all prediction depths (default: True)
    """

    def __init__(
        self,
        d_model: int,
        vocab_size: int,
        n_predict: int = 2,
        share_params: bool = True,
    ):
        super().__init__()
        self.n_predict = n_predict
        self.share_params = share_params

        if share_params:
            self.shared_layer = MTPLayer(d_model, vocab_size)
        else:
            self.layers = nn.ModuleList([MTPLayer(d_model, vocab_size) for _ in range(n_predict)])

    def _get_layer(self, depth: int) -> MTPLayer:
        if self.share_params:
            return self.shared_layer
        return self.layers[depth]

    def forward(self, h: Tensor) -> list[Tensor]:
        """Compute MTP losses for each prediction depth.

        Args:
            h: (B, T, d_model) — final hidden states from backbone.

        Returns:
            List of logit tensors, one per prediction depth, each (B, T, vocab_size).
        """
        B, T, D = h.shape
        all_logits: list[Tensor] = []
        h_prev = h

        for depth in range(self.n_predict):
            layer = self._get_layer(depth)
            h_prev, logits = layer(h, h_prev)
            all_logits.append(logits)

        return all_logits

    def compute_loss(self, h: Tensor, labels: Tensor) -> Tensor:
        """Compute MTP auxiliary loss.

        Args:
            h: (B, T, d_model) — final hidden states.
            labels: (B, T + n_predict) — target token ids, shifted.

        Returns:
            Scalar MTP loss (mean over all prediction depths and tokens).
        """
        all_logits = self.forward(h)
        total_loss = 0.0

        for depth, logits in enumerate(all_logits):
            shift = depth + 1
            if shift >= labels.shape[1]:
                break
            target = labels[:, shift : shift + logits.shape[1]]
            if target.shape[1] > logits.shape[1]:
                target = target[:, : logits.shape[1]]
            elif target.shape[1] < logits.shape[1]:
                logits = logits[:, : target.shape[1]]
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                target.reshape(-1),
            )
            total_loss = total_loss + loss

        return total_loss / self.n_predict

    @torch.no_grad()
    def draft_tokens(self, h: Tensor, temperature: float = 1.0) -> Tensor:
        """Generate draft tokens for speculative decoding.

        Args:
            h: (B, T, d_model) — final hidden states.
            temperature: Sampling temperature.

        Returns:
            (B, n_predict) — draft token ids.
        """
        if self.n_predict == 0:
            return torch.empty(h.shape[0], 0, dtype=torch.long, device=h.device)

        B = h.shape[0]

        h_prev = h[:, -1:, :]
        draft_embeds = h_prev
        drafts: list[Tensor] = []

        for depth in range(self.n_predict):
            layer = self._get_layer(depth)

            combined = torch.cat(
                [draft_embeds, h_prev.expand(-1, draft_embeds.shape[1], -1)], dim=-1
            )
            normed = layer.norm(F.silu(layer.linear(combined)))
            logits = layer.head(normed)

            if logits.shape[1] > 1:
                logits = logits[:, -1:, :]

            probs = F.softmax(logits / temperature, dim=-1)
            token = torch.multinomial(probs.view(B, -1), 1)
            drafts.append(token)

            token_embed = (
                layer.head.weight[token].unsqueeze(0) if layer.head.weight.ndim == 2 else None
            )
            if token_embed is None:
                break
            draft_embeds = token_embed

        return torch.cat(drafts, dim=-1)
