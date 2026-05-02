import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from memory_core import AurelianMemoryCore
from nn_utils import RMSNorm, RotaryEmbedding, apply_rotary, FeedForward, sample_with_top_p_top_k


class FlashAttention(nn.Module):
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
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(out.transpose(1, 2).reshape(b, t, d))


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, d_mem: int,
                 episodic_slots: int, lts_capacity: int, layer_idx: int):
        super().__init__()
        self.attn = FlashAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)
        self.memory = AurelianMemoryCore(
            d_model=d_model,
            d_mem=d_mem,
            episodic_slots=episodic_slots,
            lts_capacity=lts_capacity,
            consolidation_freq=128,
        )
        self.gate_mem = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, mask)
        x = x + self.ffn(self.norm2(x))
        mem_out = self.memory(self.norm3(x))
        x = x + torch.tanh(self.gate_mem) * mem_out
        return x


class AureliusModel1B(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = False

        self.token_embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.rotary = RotaryEmbedding(config['d_model'] // config['n_heads'],
                                      max_position=config['max_seq_len'])
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

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.ndim >= 2 and 'memory' not in name:
                nn.init.normal_(p, mean=0.0, std=0.02)
            elif 'memory' in name and p.ndim >= 2:
                nn.init.xavier_uniform_(p, gain=0.5)

    def forward(self, input_ids: torch.Tensor,
                return_mem_state: bool = False,
                return_hidden: bool = False) -> torch.Tensor | tuple[torch.Tensor, dict]:
        b, t = input_ids.shape
        h = self.token_embedding(input_ids)
        cos, sin = self.rotary(h)

        mem_states = {}
        for i, block in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    block, h, cos, sin, causal_mask,
                    use_reentrant=False,
                )
            else:
                h = block(h, cos, sin, causal_mask)
            if return_mem_state:
                _, ms = block.memory(h, return_mem_state=True)
                mem_states[f'layer_{i}'] = ms

        h = self.norm_out(h)
        logits = self.lm_head(h)

        if return_hidden:
            return logits, h
        if return_mem_state:
            return logits, mem_states
        return logits

    def _generate_with_cache(self, input_ids: torch.Tensor, kv_cache: list):
        # KV-cache optimization not yet implemented in FlashAttention;
        # fallback to full forward pass for correctness.
        return self(input_ids)

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 100,
                 temperature: float = 1.0, top_k: int = 50,
                 use_kv_cache: bool = True) -> torch.Tensor:
        self.eval()
        for step in range(max_new_tokens):
            logits = self(input_ids[:, -self.config['max_seq_len']:])
            next_token = sample_with_top_p_top_k(logits[:, -1, :], temperature, top_k)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids
