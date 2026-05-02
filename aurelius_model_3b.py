import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from memory_core import AurelianMemoryCore
from nn_utils import RMSNorm, RotaryEmbedding, FeedForward, sample_with_top_p_top_k


class FlashAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(out.transpose(1, 2).reshape(b, t, d))


class AgentSkillBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mem: int,
                 episodic_slots: int, lts_capacity: int,
                 skill_dim: int = 256, n_known_tools: int = 128):
        super().__init__()
        self.attn = FlashAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff=d_model * 4)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.memory = AurelianMemoryCore(
            d_model=d_model, d_mem=d_mem,
            episodic_slots=episodic_slots,
            lts_capacity=lts_capacity,
        )
        self.norm3 = RMSNorm(d_model)
        self.gate_mem = nn.Parameter(torch.zeros(1))

        from agent_core import ToolFormerAdapter
        from skills import SkillLibrary
        self.tool_adapter = ToolFormerAdapter(d_model, n_heads, n_known_tools)
        self.skill_lib = SkillLibrary(d_model, skill_dim)
        self.norm4 = RMSNorm(d_model)
        self.gate_agent = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                tool_descs: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.ffn(self.norm2(x))
        mem_out = self.memory(self.norm3(x))
        x = x + torch.tanh(self.gate_mem) * mem_out
        h_agent, _ = self.tool_adapter(self.norm4(x), tool_descs)
        h_skill, _ = self.skill_lib(h_agent)
        x = x + torch.tanh(self.gate_agent) * h_skill
        return x


class AureliusModel3B(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config

        self.token_embedding = nn.Embedding(config['vocab_size'], config['d_model'])
        self.rotary = RotaryEmbedding(
            config['d_model'] // config['n_heads'],
            max_position=config['max_seq_len'],
        )
        self.blocks = nn.ModuleList([
            AgentSkillBlock(
                d_model=config['d_model'],
                n_heads=config['n_heads'],
                d_mem=config['d_mem'],
                episodic_slots=config['episodic_slots'],
                lts_capacity=config['lts_capacity'],
                skill_dim=config.get('skill_dim', 256),
                n_known_tools=config.get('n_known_tools', 128),
            ) for _ in range(config['n_layers'])
        ])
        self.norm_out = RMSNorm(config['d_model'])
        self.lm_head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)
        self.token_embedding.weight = self.lm_head.weight

        from agent_loop import AgentLoopController, AgentMemoryBridge
        self.agent_controller = AgentLoopController(
            d_model=config['d_model'],
            n_heads=config['n_heads'],
            d_mem=config['d_mem'],
            n_known_tools=config.get('n_known_tools', 128),
            n_simulations=config.get('n_simulations', 16),
        )
        self.memory_bridge = AgentMemoryBridge(
            d_model=config['d_model'], d_mem=config['d_mem'],
            episodic_slots=config['episodic_slots'],
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.ndim >= 2 and 'memory' not in name and 'skill' not in name and 'tool' not in name:
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * self.config['n_layers']) ** 0.5)
            elif p.ndim >= 2:
                nn.init.xavier_uniform_(p, gain=0.5)

    def forward(self, input_ids: torch.Tensor,
                tool_descs: torch.Tensor | None = None,
                return_agent_state: bool = False) -> dict:
        b, t = input_ids.shape
        h = self.token_embedding(input_ids)
        cos, sin = self.rotary(h)

        episodic = None
        for i, block in enumerate(self.blocks):
            if self.training and self.config.get('gradient_checkpointing'):
                h = torch.utils.checkpoint.checkpoint(
                    block, h, cos, sin, tool_descs,
                    use_reentrant=False,
                )
            else:
                h = block(h, cos, sin, tool_descs)
            # Collect d_mem episodic representations via memory core q_proj
            mem_repr = block.memory.q_proj(h).mean(dim=1, keepdim=True)
            episodic = mem_repr if episodic is None else episodic + mem_repr
        episodic = episodic / len(self.blocks)

        h = self.norm_out(h)
        logits = self.lm_head(h)

        agent_out = self.agent_controller(h, tool_descs, full_cycle=True)
        h = self.memory_bridge.read_from_memory(h, episodic)

        if return_agent_state:
            return {'logits': logits, 'agent': agent_out, 'hidden': h}
        return {'logits': logits}

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 200,
                 temperature: float = 0.8, top_p: float = 0.9) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            out = self(input_ids[:, -self.config['max_seq_len']:])
            next_token = sample_with_top_p_top_k(out['logits'][:, -1, :], temperature, top_k=0, top_p=top_p)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids
