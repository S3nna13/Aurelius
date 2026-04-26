"""Aurelius — Tokenization-Free byte-level language model with patch encoding.

Architecture:
  - ByteEncoder: embed 256 byte values + Conv1d patch compression
  - ByteDecoder: expand patches back to per-byte logits over 256 values
  - CrossPatchAttention: local byte-level transformer + global patch-level transformer
  - ByteLevelLM: full model (encoder -> transformer layers -> decoder)
  - ByteMetrics: bits-per-byte, byte accuracy, top-k byte accuracy
  - ByteModelConfig: default hyperparameters (dataclass)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# Internal building blocks
# ---------------------------------------------------------------------------


class _LayerNorm(nn.Module):
    """Standard LayerNorm wrapper."""

    def __init__(self, d_model: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(d_model, eps=eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x)


class _SelfAttention(nn.Module):
    """Multi-head self-attention (native PyTorch only)."""

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"  # noqa: S101
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, causal: bool = True) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(C, dim=-1)

        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        scale = math.sqrt(self.head_dim)
        attn = torch.matmul(q, k.transpose(-2, -1)) / scale

        if causal:
            mask = torch.tril(torch.ones(T, T, device=x.device, dtype=torch.bool))
            attn = attn.masked_fill(~mask, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class _FFN(nn.Module):
    """Position-wise feed-forward with GELU activation."""

    def __init__(self, d_model: int, expansion: int = 4) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_model * expansion, bias=False)
        self.fc2 = nn.Linear(d_model * expansion, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class _TransformerBlock(nn.Module):
    """Pre-norm transformer block: self-attention + FFN with residuals."""

    def __init__(self, d_model: int, n_heads: int, causal: bool = True) -> None:
        super().__init__()
        self.norm1 = _LayerNorm(d_model)
        self.attn = _SelfAttention(d_model, n_heads)
        self.norm2 = _LayerNorm(d_model)
        self.ffn = _FFN(d_model)
        self.causal = causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), causal=self.causal)
        x = x + self.ffn(self.norm2(x))
        return x


# ---------------------------------------------------------------------------
# ByteEncoder
# ---------------------------------------------------------------------------


class ByteEncoder(nn.Module):
    """Embed raw bytes and compress them into patches via Conv1d.

    Args:
        d_model:    Hidden dimension.
        patch_size: Number of bytes grouped into one patch.
    """

    def __init__(self, d_model: int, patch_size: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.byte_embedding = nn.Embedding(256, d_model)
        self.patch_conv = nn.Conv1d(
            d_model,
            d_model,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            byte_ids: LongTensor [B, T_bytes], values in [0, 255]

        Returns:
            Tensor [B, T_patches, d_model]  where T_patches = T_bytes // patch_size
        """
        B, T = byte_ids.shape
        T_trunc = (T // self.patch_size) * self.patch_size
        byte_ids = byte_ids[:, :T_trunc]

        x = self.byte_embedding(byte_ids)  # [B, T_trunc, d_model]
        x = x.transpose(1, 2)  # [B, d_model, T_trunc]
        x = self.patch_conv(x)  # [B, d_model, T_patches]
        x = x.transpose(1, 2)  # [B, T_patches, d_model]
        return x


# ---------------------------------------------------------------------------
# ByteDecoder
# ---------------------------------------------------------------------------


class ByteDecoder(nn.Module):
    """Expand patch representations to byte-level logits over 256 values.

    Args:
        d_model:    Hidden dimension.
        patch_size: Number of byte positions per patch.
    """

    def __init__(self, d_model: int, patch_size: int = 4) -> None:
        super().__init__()
        self.d_model = d_model
        self.patch_size = patch_size
        self.patch_proj = nn.Linear(d_model, d_model * patch_size, bias=True)
        self.byte_head = nn.Linear(d_model, 256, bias=True)

    def forward(self, patch_repr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            patch_repr: [B, T_patches, d_model]

        Returns:
            logits: [B, T_patches * patch_size, 256]
        """
        B, T_patches, _ = patch_repr.shape
        expanded = self.patch_proj(patch_repr)  # [B, T_patches, d_model*patch_size]
        expanded = expanded.view(
            B, T_patches * self.patch_size, self.d_model
        )  # [B, T_bytes, d_model]
        logits = self.byte_head(expanded)  # [B, T_bytes, 256]
        return logits


# ---------------------------------------------------------------------------
# CrossPatchAttention
# ---------------------------------------------------------------------------


class CrossPatchAttention(nn.Module):
    """Hierarchical byte encoder combining local and global attention.

    Steps:
      1. Embed bytes.
      2. Local transformer blocks on the full byte sequence (non-causal).
      3. Mean pooling within non-overlapping patches.
      4. Global transformer blocks on the patch sequence (causal).

    Args:
        d_model:         Hidden dimension.
        n_heads:         Attention heads for both local and global blocks.
        patch_size:      Bytes per patch.
        n_local_layers:  Number of local (byte-level) transformer blocks.
        n_global_layers: Number of global (patch-level) transformer blocks.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        patch_size: int = 4,
        n_local_layers: int = 2,
        n_global_layers: int = 1,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.patch_size = patch_size

        self.byte_embedding = nn.Embedding(256, d_model)

        self.local_attn = nn.ModuleList(
            [_TransformerBlock(d_model, n_heads, causal=False) for _ in range(n_local_layers)]
        )
        self.patch_pool = None  # mean pooling applied in forward()

        self.global_attn = nn.ModuleList(
            [_TransformerBlock(d_model, n_heads, causal=True) for _ in range(n_global_layers)]
        )

        self.norm = _LayerNorm(d_model)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            byte_ids: LongTensor [B, T_bytes]

        Returns:
            patch_repr: [B, T_patches, d_model]
        """
        B, T = byte_ids.shape
        T_trunc = (T // self.patch_size) * self.patch_size
        byte_ids = byte_ids[:, :T_trunc]

        x = self.byte_embedding(byte_ids)  # [B, T_trunc, d_model]

        for blk in self.local_attn:
            x = blk(x)

        T_patches = T_trunc // self.patch_size
        x = x.view(B, T_patches, self.patch_size, self.d_model).mean(
            dim=2
        )  # [B, T_patches, d_model]

        for blk in self.global_attn:
            x = blk(x)

        x = self.norm(x)
        return x


# ---------------------------------------------------------------------------
# ByteLevelLM
# ---------------------------------------------------------------------------


class ByteLevelLM(nn.Module):
    """Full byte-level language model.

    Pipeline:
      byte_ids -> ByteEncoder -> transformer_layers -> ByteDecoder -> logits

    Args:
        d_model:    Hidden dimension.
        n_layers:   Number of transformer blocks on the patch sequence.
        patch_size: Bytes per patch.
        n_heads:    Number of attention heads.
    """

    def __init__(
        self,
        d_model: int,
        n_layers: int,
        patch_size: int = 4,
        n_heads: int = 4,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.patch_size = patch_size
        self.n_heads = n_heads

        self.encoder = ByteEncoder(d_model=d_model, patch_size=patch_size)
        self.transformer_layers = nn.ModuleList(
            [_TransformerBlock(d_model, n_heads, causal=True) for _ in range(n_layers)]
        )
        self.norm = _LayerNorm(d_model)
        self.decoder = ByteDecoder(d_model=d_model, patch_size=patch_size)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            byte_ids: LongTensor [B, T_bytes]

        Returns:
            logits: [B, T_bytes_trunc, 256]
              where T_bytes_trunc = (T_bytes // patch_size) * patch_size
        """
        patch_repr = self.encoder(byte_ids)
        x = patch_repr
        for blk in self.transformer_layers:
            x = blk(x)
        x = self.norm(x)
        logits = self.decoder(x)
        return logits

    def compute_loss(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """Causal LM loss: cross-entropy on next-byte prediction.

        Predicts byte at position t+1 from context at positions 0..t.

        Args:
            byte_ids: LongTensor [B, T_bytes]

        Returns:
            Scalar cross-entropy loss.
        """
        logits = self.forward(byte_ids)  # [B, T_trunc, 256]
        T_trunc = logits.shape[1]

        targets = byte_ids[:, 1 : T_trunc + 1]  # [B, T_trunc] (may be shorter)
        if targets.shape[1] < T_trunc:
            pad_len = T_trunc - targets.shape[1]
            targets = F.pad(targets, (0, pad_len), value=0)

        loss = F.cross_entropy(
            logits.reshape(-1, 256),
            targets.reshape(-1).long(),
        )
        return loss

    @torch.no_grad()
    def generate_bytes(
        self,
        prefix_bytes: torch.Tensor,
        max_new: int,
    ) -> torch.Tensor:
        """Greedy autoregressive byte generation.

        Args:
            prefix_bytes: LongTensor [T_prefix]
            max_new:      Number of new bytes to generate.

        Returns:
            LongTensor [T_prefix + max_new]
        """
        self.eval()
        device = next(self.parameters()).device
        seq = prefix_bytes.to(device)

        for _ in range(max_new):
            if seq.shape[0] < self.patch_size:
                pad = torch.zeros(self.patch_size - seq.shape[0], dtype=torch.long, device=device)
                inp = torch.cat([pad, seq], dim=0)
            else:
                inp = seq

            inp_batch = inp.unsqueeze(0)  # [1, T]
            logits = self.forward(inp_batch)  # [1, T_trunc, 256]
            next_byte = logits[0, -1, :].argmax(dim=-1).unsqueeze(0)  # [1]
            seq = torch.cat([seq, next_byte], dim=0)

        return seq


# ---------------------------------------------------------------------------
# ByteMetrics
# ---------------------------------------------------------------------------


class ByteMetrics:
    """Evaluation metrics for byte-level language models."""

    @staticmethod
    def bits_per_byte(model: ByteLevelLM, byte_ids: torch.Tensor) -> float:
        """Bits-per-byte (BPB) metric.

        BPB = mean_CE_loss / log(2).
        A random uniform model yields BPB = log2(256) = 8.

        Args:
            model:    ByteLevelLM in eval mode.
            byte_ids: LongTensor [B, T_bytes]

        Returns:
            BPB as a positive Python float.
        """
        model.eval()
        with torch.no_grad():
            loss = model.compute_loss(byte_ids)
        return (loss / math.log(2)).item()

    @staticmethod
    def byte_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
        """Top-1 byte prediction accuracy.

        Args:
            logits:  [B, T, 256]
            targets: [B, T]  LongTensor in [0, 255]

        Returns:
            Accuracy in [0, 1].
        """
        preds = logits.argmax(dim=-1)
        return (preds == targets).float().mean().item()

    @staticmethod
    def top_k_byte_accuracy(
        logits: torch.Tensor,
        targets: torch.Tensor,
        k: int = 5,
    ) -> float:
        """Top-k byte prediction accuracy.

        Args:
            logits:  [B, T, 256]
            targets: [B, T]  LongTensor in [0, 255]
            k:       Number of top predictions considered correct.

        Returns:
            Top-k accuracy in [0, 1].
        """
        topk = logits.topk(k, dim=-1).indices  # [B, T, k]
        targets_exp = targets.unsqueeze(-1).expand_as(topk)
        return (topk == targets_exp).any(dim=-1).float().mean().item()


# ---------------------------------------------------------------------------
# ByteModelConfig
# ---------------------------------------------------------------------------


@dataclass
class ByteModelConfig:
    """Default hyperparameters for the byte-level LM."""

    d_model: int = 32
    n_layers: int = 2
    patch_size: int = 4
    n_heads: int = 4
    n_local_layers: int = 1
    n_global_layers: int = 1
