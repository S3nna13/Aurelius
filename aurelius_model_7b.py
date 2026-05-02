import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Dict, Any

from memory_core import AurelianMemoryCore
from agent_core import ToolFormerAdapter, PlanningModule, CriticHead, ValueHead
from agent_loop import AgentLoopController, AgentMemoryBridge, ExperienceReplayBuffer
from skills import SkillLibrary
from nn_utils import RMSNorm, FeedForward, sample_with_top_p_top_k


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position: int = 16384, theta: float = 10000.0):
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)

    def forward(self, x: torch.Tensor, offset: int = 0) -> tuple:
        t = torch.arange(offset, offset + x.shape[1], device=x.device).float()
        freqs = t[:, None] @ self.inv_freq[None, :]
        cos = freqs.cos().unsqueeze(0).unsqueeze(1)
        sin = freqs.sin().unsqueeze(0).unsqueeze(1)
        return cos, sin


def apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos[..., :d] - x2 * sin[..., :d],
                      x1 * sin[..., :d] + x2 * cos[..., :d]], dim=-1)


class FlashAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q = apply_rotary(q, cos, sin)
        k = apply_rotary(k, cos, sin)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.out(out.transpose(1, 2).reshape(b, t, d))


class AureliusBlock7B(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, d_mem: int,
                 episodic_slots: int, lts_capacity: int,
                 skill_dim: int = 512, n_known_tools: int = 256,
                 layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx

        self.attn = FlashAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model, d_ff)
        self.norm1 = RMSNorm(d_model)
        self.norm2 = RMSNorm(d_model)
        self.norm3 = RMSNorm(d_model)
        self.gate_mem = nn.Parameter(torch.zeros(1))

        self.memory = AurelianMemoryCore(
            d_model=d_model, d_mem=d_mem,
            episodic_slots=episodic_slots,
            lts_capacity=lts_capacity,
            consolidation_freq=512,
        )

        if layer_idx % 4 == 3:
            self.has_agent = True
            self.tool_adapter = ToolFormerAdapter(d_model, n_heads, n_known_tools)
            self.skill_lib = SkillLibrary(d_model, skill_dim)
            self.norm4 = RMSNorm(d_model)
            self.gate_agent = nn.Parameter(torch.zeros(1))
        else:
            self.has_agent = False

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                tool_descs: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), cos, sin, mask)
        x = x + self.ffn(self.norm2(x))
        mem_out = self.memory(self.norm3(x))
        x = x + torch.tanh(self.gate_mem) * mem_out
        if self.has_agent:
            h_agent, _ = self.tool_adapter(self.norm4(x), tool_descs)
            h_skill, _ = self.skill_lib(h_agent)
            x = x + torch.tanh(self.gate_agent) * h_skill
        return x


class BrainBridge(nn.Module):
    def __init__(self, d_model: int, d_brain: int = 1024, n_actions: int = 8):
        super().__init__()
        self.proj_in = nn.Linear(d_model, d_brain)
        self.proj_out = nn.Linear(d_brain, d_model)
        self.gate = nn.Parameter(torch.zeros(1))

        self.planner = PlanningModule(d_brain, n_simulations=8, max_depth=4)
        self.critic = CriticHead(d_brain)
        self.uncertainty = nn.Sequential(
            nn.Linear(d_brain, d_brain // 4),
            nn.ReLU(),
            nn.Linear(d_brain // 4, 2),
        )

    def forward(self, h: torch.Tensor) -> Dict[str, torch.Tensor]:
        brain_in = self.proj_in(h)
        plan, value, _ = self.planner(brain_in)
        critic_score, critic_suggestion = self.critic(
            brain_in[:, -1] if brain_in.dim() == 3 else brain_in,
            plan,
        )
        unc = self.uncertainty(brain_in[:, -1] if brain_in.dim() == 3 else brain_in)
        epi, alea = unc.chunk(2, dim=-1)
        brain_out = self.proj_out(plan.unsqueeze(1) if plan.dim() == 2 else plan)
        if brain_out.dim() == 2:
            brain_out = brain_out.unsqueeze(1)
        gate = torch.sigmoid(self.gate)
        output = h + gate * brain_out
        return {
            'output': output,
            'value': value,
            'critic_score': critic_score,
            'epistemic_uncertainty': epi.squeeze(-1),
            'aleatoric_uncertainty': alea.squeeze(-1),
        }


class AureliusModel7B(nn.Module):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.gradient_checkpointing = config.get('gradient_checkpointing', True)

        d_model = config['d_model']
        n_heads = config['n_heads']
        d_ff = config['d_ff']
        d_mem = config['d_mem']
        episodic_slots = config['episodic_slots']
        lts_capacity = config['lts_capacity']
        skill_dim = config.get('skill_dim', 512)
        n_known_tools = config.get('n_known_tools', 256)
        d_brain = config.get('d_brain', 1024)

        self.token_embedding = nn.Embedding(config['vocab_size'], d_model)
        self.rotary = RotaryEmbedding(
            d_model // n_heads,
            max_position=config['max_seq_len'],
        )

        self.blocks = nn.ModuleList([
            AureliusBlock7B(
                d_model=d_model,
                n_heads=n_heads,
                d_ff=d_ff,
                d_mem=d_mem,
                episodic_slots=episodic_slots,
                lts_capacity=lts_capacity,
                skill_dim=skill_dim,
                n_known_tools=n_known_tools,
                layer_idx=i,
            ) for i in range(config['n_layers'])
        ])

        self.norm_out = RMSNorm(d_model)
        self.lm_head = nn.Linear(d_model, config['vocab_size'], bias=False)
        self.token_embedding.weight = self.lm_head.weight

        self.brain = BrainBridge(d_model, d_brain, n_actions=config.get('n_actions', 8))

        self.agent_controller = AgentLoopController(
            d_model=d_model,
            n_heads=n_heads,
            d_mem=d_mem,
            n_known_tools=n_known_tools,
            n_simulations=config.get('n_simulations', 24),
        )
        self.memory_bridge = AgentMemoryBridge(
            d_model=d_model,
            d_mem=d_mem,
            episodic_slots=episodic_slots,
        )
        self.replay_buffer = ExperienceReplayBuffer(
            capacity=config.get('replay_capacity', 100000),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.ndim >= 2 and 'memory' not in name and 'skill' not in name and 'tool' not in name and 'brain' not in name:
                nn.init.normal_(p, mean=0.0, std=0.02 / (2 * self.config['n_layers']) ** 0.5)
            elif p.ndim >= 2:
                nn.init.xavier_uniform_(p, gain=0.5)

    def forward(self, input_ids: torch.Tensor,
                tool_descs: Optional[torch.Tensor] = None,
                return_agent_state: bool = False,
                use_brain: bool = True) -> Dict[str, Any]:
        b, t = input_ids.shape
        h = self.token_embedding(input_ids)
        cos, sin = self.rotary(h)

        episodic = None
        for i, block in enumerate(self.blocks):
            if self.gradient_checkpointing and self.training:
                h = torch.utils.checkpoint.checkpoint(
                    block, h, cos, sin, None, tool_descs,
                    use_reentrant=False,
                )
            else:
                h = block(h, cos, sin, tool_descs=tool_descs)
            mem_repr = block.memory.q_proj(h).mean(dim=1, keepdim=True)
            episodic = mem_repr if episodic is None else episodic + mem_repr
        episodic = episodic / len(self.blocks)

        h = self.norm_out(h)
        logits = self.lm_head(h)

        agent_out = self.agent_controller(h, tool_descs, full_cycle=True)
        h = self.memory_bridge.read_from_memory(h, episodic)

        if use_brain:
            brain_result = self.brain(h)
            h = brain_result['output']

        aux_losses = {}
        for i, block in enumerate(self.blocks):
            if block.has_agent:
                if hasattr(block, 'tool_adapter') and hasattr(block.tool_adapter, 'call_head'):
                    _, call = block.tool_adapter(block.norm4(h), tool_descs, return_call=True)
                    if call is not None:
                        tool_logits, _, _ = call
                        aux_losses[f'tool_entropy_l{i}'] = -(F.softmax(tool_logits, dim=-1) * F.log_softmax(tool_logits, dim=-1)).sum(-1).mean()

        if return_agent_state:
            return {
                'logits': logits,
                'agent': agent_out,
                'hidden': h,
                'brain': brain_result if use_brain else None,
                'aux_losses': aux_losses,
                'episodic': episodic,
            }
        return {'logits': logits}

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, max_new_tokens: int = 200,
                 temperature: float = 0.8, top_p: float = 0.9,
                 use_brain: bool = False) -> torch.Tensor:
        self.eval()
        for _ in range(max_new_tokens):
            out = self(input_ids[:, -self.config['max_seq_len']:], use_brain=use_brain)
            next_token = sample_with_top_p_top_k(out['logits'][:, -1, :], temperature, top_k=0, top_p=top_p)
            input_ids = torch.cat([input_ids, next_token], dim=-1)
        return input_ids

    def count_parameters(self) -> Dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        memory = sum(p.numel() for n, p in self.named_parameters() if 'memory' in n)
        agent = sum(p.numel() for n, p in self.named_parameters() if 'tool' in n or 'skill' in n or 'agent' in n)
        brain = sum(p.numel() for n, p in self.named_parameters() if 'brain' in n)
        return {
            'total': total,
            'trainable': trainable,
            'memory': memory,
            'agent': agent,
            'brain': brain,
            'transformer': total - memory - agent - brain,
        }

    @classmethod
    def from_config_file(cls, path: str) -> 'AureliusModel7B':
        import yaml
        with open(path) as f:
            config = yaml.safe_load(f)
        key = list(config.keys())[0]
        model_config = config[key]
        training_config = config.get('training', {})
        model_config.update({
            'gradient_checkpointing': training_config.get('gradient_checkpointing', True),
            'd_brain': training_config.get('d_brain', 1024),
            'n_actions': training_config.get('n_actions', 8),
        })
        model_config.update(config.get('data', {}))
        return cls(model_config)