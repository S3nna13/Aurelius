import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class ImportanceScorer(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(d_model * 3, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, key: torch.Tensor, value: torch.Tensor, hidden_state: torch.Tensor) -> torch.Tensor:
        b, h, n, d_head = key.shape
        d_model = h * d_head
        k_flat = key.permute(0, 2, 1, 3).reshape(b, n, d_model)
        v_flat = value.permute(0, 2, 1, 3).reshape(b, n, d_model)
        if hidden_state.dim() == 2:
            hidden_state = hidden_state.unsqueeze(1)
        if hidden_state.shape[1] == 1:
            hidden_state = hidden_state.expand(b, n, -1)
        x = torch.cat([k_flat, v_flat, hidden_state], dim=-1)
        x = F.relu(self.fc1(x))
        return self.fc2(x).squeeze(-1)


class LearnedEvictionPolicy(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int = 256):
        super().__init__()
        self.scorer = ImportanceScorer(d_model, hidden_dim)

    def evict(
        self, k: torch.Tensor, v: torch.Tensor, scores: torch.Tensor, n_evict: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        b, h, n, d = k.shape
        keep = n - n_evict
        if n_evict <= 0:
            empty_k = k.new_empty(0)
            empty_v = v.new_empty(0)
            empty_s = scores.new_empty(0)
            return k, v, scores, empty_k, empty_v, empty_s
        sort_idx = scores.argsort(dim=1, descending=True)
        sort_idx_exp = sort_idx[:, None, :, None].expand(-1, h, -1, d)
        k_sorted = torch.gather(k, 2, sort_idx_exp)
        v_sorted = torch.gather(v, 2, sort_idx_exp)
        scores_sorted = torch.gather(scores, 1, sort_idx)
        keep_k = k_sorted[:, :, :keep]
        keep_v = v_sorted[:, :, :keep]
        keep_scores = scores_sorted[:, :keep]
        evict_k = k_sorted[:, :, keep:]
        evict_v = v_sorted[:, :, keep:]
        evict_scores = scores_sorted[:, keep:]
        return keep_k, keep_v, keep_scores, evict_k, evict_v, evict_scores


class HierarchicalKVCache(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        head_dim: int,
        cap1: int = 512,
        cap2: int = 2048,
        cap3: int = 4096,
        max_batch: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.cap1 = cap1
        self.cap2 = cap2
        self.cap3 = cap3

        self.register_buffer('t1_k', torch.zeros(max_batch, n_heads, cap1, head_dim))
        self.register_buffer('t1_v', torch.zeros(max_batch, n_heads, cap1, head_dim))
        self.register_buffer('t1_scores', torch.zeros(max_batch, cap1))
        self.register_buffer('t1_n', torch.zeros(max_batch, dtype=torch.long))

        self.register_buffer('t2_k', torch.zeros(max_batch, n_heads, cap2, head_dim, dtype=torch.half))
        self.register_buffer('t2_v', torch.zeros(max_batch, n_heads, cap2, head_dim, dtype=torch.half))
        self.register_buffer('t2_scores', torch.zeros(max_batch, cap2, dtype=torch.half))
        self.register_buffer('t2_n', torch.zeros(max_batch, dtype=torch.long))

        self.register_buffer('t3_k', torch.zeros(max_batch, n_heads, cap3, head_dim, dtype=torch.int8))
        self.register_buffer('t3_v', torch.zeros(max_batch, n_heads, cap3, head_dim, dtype=torch.int8))
        self.register_buffer('t3_s_k', torch.ones(max_batch, n_heads, cap3, 1))
        self.register_buffer('t3_s_v', torch.ones(max_batch, n_heads, cap3, 1))
        self.register_buffer('t3_scores', torch.zeros(max_batch, cap3, dtype=torch.half))
        self.register_buffer('t3_n', torch.zeros(max_batch, dtype=torch.long))

        self.eviction_policy = LearnedEvictionPolicy(d_model)

    def _quantize(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-5) / 127.0
        q = (x / scale).round().clamp(-128, 127).to(torch.int8)
        return q, scale

    def _dequantize(self, q: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return q.float() * scale

    def write(self, key: torch.Tensor, value: torch.Tensor, hidden_state: torch.Tensor):
        b, h, n_new, d = key.shape
        if b != self.t1_k.shape[0]:
            raise ValueError(f"Batch mismatch: expected {self.t1_k.shape[0]}, got {b}")
        new_scores = self.eviction_policy.scorer(key, value, hidden_state)

        t1_n = self.t1_n[0].item()
        need_evict = max(0, t1_n + n_new - self.cap1)

        if need_evict > 0:
            n_keep_old = max(0, self.cap1 - n_new)
            old_scores = self.t1_scores[:, :t1_n]
            sorted_old = old_scores.argsort(dim=1, descending=True)
            keep_old_idx = sorted_old[:, :n_keep_old]
            evict_old_idx = sorted_old[:, n_keep_old:]

            keep_idx_exp = keep_old_idx[:, None, :, None].expand(-1, h, -1, d)
            kept_k = torch.gather(self.t1_k[:, :, :t1_n], 2, keep_idx_exp)
            kept_v = torch.gather(self.t1_v[:, :, :t1_n], 2, keep_idx_exp)
            kept_scores = torch.gather(self.t1_scores[:, :t1_n], 1, keep_old_idx)

            evict_idx_exp = evict_old_idx[:, None, :, None].expand(-1, h, -1, d)
            evict_k = torch.gather(self.t1_k[:, :, :t1_n], 2, evict_idx_exp)
            evict_v = torch.gather(self.t1_v[:, :, :t1_n], 2, evict_idx_exp)
            evict_scores = torch.gather(self.t1_scores[:, :t1_n], 1, evict_old_idx)

            self.t1_k[:, :, :n_keep_old] = kept_k
            self.t1_v[:, :, :n_keep_old] = kept_v
            self.t1_scores[:, :n_keep_old] = kept_scores
            self.t1_k[:, :, n_keep_old:self.cap1] = key
            self.t1_v[:, :, n_keep_old:self.cap1] = value
            self.t1_scores[:, n_keep_old:self.cap1] = new_scores
            self.t1_n.fill_(self.cap1)

            self._evict_to_tier2(evict_k, evict_v, evict_scores)
        else:
            self.t1_k[:, :, t1_n:t1_n + n_new] = key
            self.t1_v[:, :, t1_n:t1_n + n_new] = value
            self.t1_scores[:, t1_n:t1_n + n_new] = new_scores
            self.t1_n.fill_(t1_n + n_new)

    def _evict_to_tier2(self, k: torch.Tensor, v: torch.Tensor, scores: torch.Tensor):
        b, h, n_evict, d = k.shape
        t2_n = self.t2_n[0].item()
        need_evict = max(0, t2_n + n_evict - self.cap2)

        k_half = k.half()
        v_half = v.half()
        scores_half = scores.half()

        if need_evict > 0:
            n_keep_old = max(0, self.cap2 - n_evict)
            old_scores = self.t2_scores[:, :t2_n]
            sorted_old = old_scores.argsort(dim=1, descending=True)
            keep_old_idx = sorted_old[:, :n_keep_old]
            evict_old_idx = sorted_old[:, n_keep_old:]

            keep_idx_exp = keep_old_idx[:, None, :, None].expand(-1, h, -1, d)
            kept_k = torch.gather(self.t2_k[:, :, :t2_n], 2, keep_idx_exp)
            kept_v = torch.gather(self.t2_v[:, :, :t2_n], 2, keep_idx_exp)
            kept_scores = torch.gather(self.t2_scores[:, :t2_n], 1, keep_old_idx)

            evict_old_idx_exp = evict_old_idx[:, None, :, None].expand(-1, h, -1, d)
            evict_t3_k = torch.gather(self.t2_k[:, :, :t2_n], 2, evict_old_idx_exp)
            evict_t3_v = torch.gather(self.t2_v[:, :, :t2_n], 2, evict_old_idx_exp)
            evict_t3_scores = torch.gather(self.t2_scores[:, :t2_n], 1, evict_old_idx)

            self.t2_k[:, :, :n_keep_old] = kept_k
            self.t2_v[:, :, :n_keep_old] = kept_v
            self.t2_scores[:, :n_keep_old] = kept_scores
            self.t2_k[:, :, n_keep_old:self.cap2] = k_half
            self.t2_v[:, :, n_keep_old:self.cap2] = v_half
            self.t2_scores[:, n_keep_old:self.cap2] = scores_half
            self.t2_n.fill_(self.cap2)

            self._evict_to_tier3(evict_t3_k.float(), evict_t3_v.float(), evict_t3_scores.float())
        else:
            self.t2_k[:, :, t2_n:t2_n + n_evict] = k_half
            self.t2_v[:, :, t2_n:t2_n + n_evict] = v_half
            self.t2_scores[:, t2_n:t2_n + n_evict] = scores_half
            self.t2_n.fill_(t2_n + n_evict)

    def _evict_to_tier3(self, k: torch.Tensor, v: torch.Tensor, scores: torch.Tensor):
        b, h, n_evict, d = k.shape
        t3_n = self.t3_n[0].item()
        need_evict = max(0, t3_n + n_evict - self.cap3)

        k_q, k_scale = self._quantize(k)
        v_q, v_scale = self._quantize(v)
        scores_half = scores.half()

        if need_evict > 0:
            n_keep_old = max(0, self.cap3 - n_evict)
            old_scores = self.t3_scores[:, :t3_n]
            sorted_old = old_scores.argsort(dim=1, descending=True)
            keep_old_idx = sorted_old[:, :n_keep_old]
            evict_old_idx = sorted_old[:, n_keep_old:]

            keep_idx_exp = keep_old_idx[:, None, :, None].expand(-1, h, -1, d)
            keep_sk_exp = keep_old_idx[:, None, :, None].expand(-1, h, -1, 1)
            kept_k = torch.gather(self.t3_k[:, :, :t3_n], 2, keep_idx_exp)
            kept_v = torch.gather(self.t3_v[:, :, :t3_n], 2, keep_idx_exp)
            kept_sk = torch.gather(self.t3_s_k[:, :, :t3_n], 2, keep_sk_exp)
            kept_sv = torch.gather(self.t3_s_v[:, :, :t3_n], 2, keep_sk_exp)
            kept_scores = torch.gather(self.t3_scores[:, :t3_n], 1, keep_old_idx)

            self.t3_k[:, :, :n_keep_old] = kept_k
            self.t3_v[:, :, :n_keep_old] = kept_v
            self.t3_s_k[:, :, :n_keep_old] = kept_sk
            self.t3_s_v[:, :, :n_keep_old] = kept_sv
            self.t3_scores[:, :n_keep_old] = kept_scores
            self.t3_k[:, :, n_keep_old:self.cap3] = k_q
            self.t3_v[:, :, n_keep_old:self.cap3] = v_q
            self.t3_s_k[:, :, n_keep_old:self.cap3] = k_scale
            self.t3_s_v[:, :, n_keep_old:self.cap3] = v_scale
            self.t3_scores[:, n_keep_old:self.cap3] = scores_half
            self.t3_n.fill_(self.cap3)
        else:
            self.t3_k[:, :, t3_n:t3_n + n_evict] = k_q
            self.t3_v[:, :, t3_n:t3_n + n_evict] = v_q
            self.t3_s_k[:, :, t3_n:t3_n + n_evict] = k_scale
            self.t3_s_v[:, :, t3_n:t3_n + n_evict] = v_scale
            self.t3_scores[:, t3_n:t3_n + n_evict] = scores_half
            self.t3_n.fill_(t3_n + n_evict)

    def read(self, query: Optional[torch.Tensor] = None, n_tokens: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        b = self.t1_k.shape[0]
        t1_n = self.t1_n[0].item()
        t2_n = self.t2_n[0].item()
        t3_n = self.t3_n[0].item()

        k_parts = []
        v_parts = []
        tier_labels = []

        if t3_n > 0:
            t3_k_fp = self._dequantize(self.t3_k[:, :, :t3_n], self.t3_s_k[:, :, :t3_n])
            t3_v_fp = self._dequantize(self.t3_v[:, :, :t3_n], self.t3_s_v[:, :, :t3_n])
            k_parts.append(t3_k_fp)
            v_parts.append(t3_v_fp)
            tier_labels.append(torch.full((b, t3_n), 2, device=self.t3_k.device, dtype=torch.long))

        if t2_n > 0:
            k_parts.append(self.t2_k[:, :, :t2_n].float())
            v_parts.append(self.t2_v[:, :, :t2_n].float())
            tier_labels.append(torch.full((b, t2_n), 1, device=self.t2_k.device, dtype=torch.long))

        if t1_n > 0:
            k_parts.append(self.t1_k[:, :, :t1_n])
            v_parts.append(self.t1_v[:, :, :t1_n])
            tier_labels.append(torch.zeros(b, t1_n, device=self.t1_k.device, dtype=torch.long))

        if not k_parts:
            empty = torch.zeros(b, self.n_heads, 0, self.head_dim, device=self.t1_k.device)
            return empty, empty, torch.zeros(b, 0, device=self.t1_k.device, dtype=torch.long)

        k = torch.cat(k_parts, dim=2)
        v = torch.cat(v_parts, dim=2)
        labels = torch.cat(tier_labels, dim=1)

        if n_tokens is not None and n_tokens < k.shape[2]:
            k = k[:, :, -n_tokens:]
            v = v[:, :, -n_tokens:]
            labels = labels[:, -n_tokens:]

        return k, v, labels

    def read_tier(self, tier: int) -> Tuple[torch.Tensor, torch.Tensor]:
        if tier == 1:
            n = self.t1_n[0].item()
            return self.t1_k[:, :, :n], self.t1_v[:, :, :n]
        elif tier == 2:
            n = self.t2_n[0].item()
            return self.t2_k[:, :, :n].float(), self.t2_v[:, :, :n].float()
        elif tier == 3:
            n = self.t3_n[0].item()
            k = self._dequantize(self.t3_k[:, :, :n], self.t3_s_k[:, :, :n])
            v = self._dequantize(self.t3_v[:, :, :n], self.t3_s_v[:, :, :n])
            return k, v
        raise ValueError(f'Invalid tier: {tier}')

    def get_kv_for_layer(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        k, v, labels = self.read()
        return k, v, labels

    def reset(self):
        self.t1_n.zero_()
        self.t2_n.zero_()
        self.t3_n.zero_()
        self.t1_k.zero_()
        self.t1_v.zero_()
        self.t1_scores.zero_()
        self.t2_k.zero_()
        self.t2_v.zero_()
        self.t2_scores.zero_()
        self.t3_k.zero_()
        self.t3_v.zero_()
        self.t3_scores.zero_()
        self.t3_s_k.fill_(1.0)
        self.t3_s_v.fill_(1.0)


class MultiScaleAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.tier_bias = nn.Parameter(torch.zeros(3))
        self.q_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        cache_output: Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
        causal_mask: bool = True,
        padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        k, v, tier_labels = cache_output
        b, n_q, _ = x.shape

        q = self.q_proj(x)
        q = q.view(b, n_q, self.n_heads, self.head_dim).transpose(1, 2)

        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = attn + self.tier_bias[tier_labels].unsqueeze(1).unsqueeze(2)

        if causal_mask:
            n_kv = k.shape[2]
            mask = torch.triu(
                torch.full((n_q, n_kv), float('-inf'), dtype=x.dtype, device=x.device),
                diagonal=n_kv - n_q + 1,
            )
            attn = attn + mask

        if padding_mask is not None:
            attn = attn.masked_fill(padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, n_q, self.d_model)
        out = self.out_proj(out)

        return out
