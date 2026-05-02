import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging

logger = logging.getLogger(__name__)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position: int = 4096, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_position = max_position
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, x: torch.Tensor, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = x.shape[1]
        max_len = offset + seq_len
        if self._cos_cached is None or self._cos_cached.size(-2) < max_len or self._cos_cached.device != x.device:
            t = torch.arange(max_len, device=x.device).float()
            freqs = t[:, None] @ self.inv_freq[None, :]
            self._cos_cached = freqs.cos().unsqueeze(0).unsqueeze(1)
            self._sin_cached = freqs.sin().unsqueeze(0).unsqueeze(1)
        return self._cos_cached[:, :, offset:offset + seq_len], self._sin_cached[:, :, offset:offset + seq_len]


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos[..., :d] - x2 * sin[..., :d],
                      x1 * sin[..., :d] + x2 * cos[..., :d]], dim=-1)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_ff, bias=False)
        self.up = nn.Linear(d_model, d_ff, bias=False)
        self.down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


def create_causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    mask = torch.tril(torch.ones(seq_len, seq_len, device=device))
    return mask.view(1, 1, seq_len, seq_len)


class CausalMaskCache:
    def __init__(self):
        self._mask = None
        self._seq_len = 0

    def get(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self._seq_len < seq_len or self._mask is None:
            self._mask = create_causal_mask(seq_len, device)
            self._seq_len = seq_len
        elif self._mask.device != device:
            self._mask = self._mask.to(device)
        return self._mask[:, :, :seq_len, :seq_len] if self._seq_len > seq_len else self._mask


def sample_with_top_p_top_k(
    logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int = 0,
    top_p: float = 0.0,
) -> torch.Tensor:
    logits = logits / max(temperature, 1e-8)
    if top_k > 0:
        vals, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        logits[logits < vals[:, -1:]] = float('-inf')
    if top_p > 0.0:
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        sorted_mask = cumulative_probs > top_p
        sorted_mask[..., 1:] = sorted_mask[..., :-1].clone()
        sorted_mask[..., 0] = False
        indices_to_remove = sorted_mask.scatter(1, sorted_indices, sorted_mask)
        logits[indices_to_remove] = float('-inf')
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def validate_input_ids(input_ids: torch.Tensor, vocab_size: int) -> None:
    if input_ids.dtype != torch.long:
        raise TypeError(f"input_ids must be torch.long, got {input_ids.dtype}")
    if input_ids.dim() != 2:
        raise ValueError(f"input_ids must be 2D (batch, seq), got {input_ids.dim()}D")
    if input_ids.numel() == 0:
        raise ValueError("input_ids must not be empty")
    if input_ids.max() >= vocab_size or input_ids.min() < 0:
        raise ValueError(f"input_ids values out of range [0, {vocab_size})")
