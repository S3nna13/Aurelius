import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from memory_core import AurelianMemoryCore
from nn_utils import RMSNorm, RotaryEmbedding, apply_rotary, FeedForward, sample_with_top_p_top_k


class Attention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        attn = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(attn.transpose(1, 2).reshape(b, t, d))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, d_mem: int,
                 episodic_slots: int, lts_capacity: int, layer_idx: int):
        super().__init__()
        self.attn = Attention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)
        self.memory = AurelianMemoryCore(
            d_model=d_model,
            d_mem=d_mem,
            episodic_slots=episodic_slots,
            lts_capacity=lts_capacity,
            consolidation_freq=64,
        )
        self.gate_mem = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        mem_out = self.memory(self.norm3(x))
        x = x + torch.tanh(self.gate_mem) * mem_out
        return x


class AureliusModel(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.rotary = RotaryEmbedding(config['d_model'] // config['n_heads'])
        self.blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config['d_model'],
                n_heads=config['n_heads'],
                d_ff=config['d_ff'],
                d_mem=config['d_mem'],
                episodic_slots=config['episodic_slots'],
                lts_capacity=config['lts_capacity'],
                layer_idx=i,
            ) for i in range(config['n_layers'])
        ])
        self.norm_out = RMSNorm(config['d_model'])
        self.lm_head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, input_ids: torch.Tensor,
                return_mem_state: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict]:
        b, t = input_ids.shape
        h = self.token_embedding(input_ids)
        cos, sin = self.rotary(h)

        mem_states = {}
        for i, block in enumerate(self.blocks):
            h = block(h, cos, sin)
            if return_mem_state:
                _, ms = block.memory(h, return_mem_state=True)
                mem_states[f'layer_{i}'] = ms

        h = self.norm_out(h)
        logits = self.lm_head(h)

        if return_mem_state:
            return logits, mem_states
        return logits

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: int = 50) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            logits = self(input_ids[:, -self.config['max_seq_len']:])
            next_token = sample_with_top_p_top_k(logits[:, -1, :], temperature, top_k)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids
