import torch
import torch.nn as nn
import torch.nn.functional as F
import time
from dataclasses import dataclass, field
from typing import Optional

from agent_core import ToolFormerAdapter, PlanningModule, CriticHead, ValueHead
from skills import SkillLibrary
import logging
logger = logging.getLogger(__name__)


@dataclass
class AgentStep:
    observation: torch.Tensor
    thought: str
    action: dict
    result: Optional[torch.Tensor] = None
    reward: float = 0.0
    skill_used: int = -1
    timestamp: float = 0.0


@dataclass
class AgentEpisode:
    steps: list = field(default_factory=list)
    total_reward: float = 0.0
    task_embedding: Optional[torch.Tensor] = None


class AgentLoopController(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_mem: int,
                 n_known_tools: int = 64, n_simulations: int = 8):
        super().__init__()
        self.d_model = d_model
        self.tool_adapter = ToolFormerAdapter(d_model, n_heads, n_known_tools)
        self.planner = PlanningModule(d_model, n_simulations=n_simulations)
        self.critic = CriticHead(d_model)
        self.value_head = ValueHead(d_model)
        self.skill_lib = SkillLibrary(d_model)

        self.episode_buffer: list[AgentEpisode] = []
        self.max_episodes = 100

    def observe(self, h: torch.Tensor, tool_descs: torch.Tensor | None = None) -> torch.Tensor:
        h, _ = self.tool_adapter(h, tool_descs)
        return h

    def think(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        plan, value, tree = self.planner(h)
        return plan, value

    def act(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h_return, call = self.tool_adapter(h, return_call=True)
        if call is not None:
            tool_logits, params_presence, params_raw = call
            tool_id = tool_logits.argmax(dim=-1)
            h_skill, skill_idx = self.skill_lib(h)
            return h_return, tool_id, skill_idx
        return h_return, torch.zeros(1, device=h.device), torch.zeros(1, device=h.device)

    def reflect(self, h: torch.Tensor, action: torch.Tensor) -> tuple[float, torch.Tensor]:
        score, suggestion = self.critic(h, action)
        return score.mean().item(), suggestion

    def learn(self, episode: AgentEpisode):
        self.episode_buffer.append(episode)
        if len(self.episode_buffer) > self.max_episodes:
            self.episode_buffer.pop(0)

    def forward(self, h: torch.Tensor, tool_descs: torch.Tensor | None = None,
                full_cycle: bool = True) -> dict:
        h = self.observe(h, tool_descs)
        plan, value = self.think(h)
        h_acted, tool_id, skill_idx = self.act(h)
        score, suggestion = self.reflect(h_acted, plan)
        return {
            'hidden': h_acted,
            'plan': plan,
            'value': value,
            'tool_id': tool_id,
            'skill_idx': skill_idx,
            'critic_score': score,
            'suggestion': suggestion,
        }


class AgentMemoryBridge(nn.Module):
    def __init__(self, d_model: int, d_mem: int, episodic_slots: int):
        super().__init__()
        self.d_model = d_model
        self.d_mem = d_mem
        self.episodic_slots = episodic_slots

        self.episodic_to_agent = nn.Linear(d_mem, d_model)
        self.agent_to_episodic = nn.Linear(d_model, d_mem)
        self.gate = nn.Parameter(torch.ones(1))

    def read_from_memory(self, agent_state: torch.Tensor,
                         memory_episodic: torch.Tensor) -> torch.Tensor:
        mem_proj = self.episodic_to_agent(memory_episodic)
        attn = agent_state @ mem_proj.transpose(-2, -1)
        attn = F.softmax(attn / (self.d_model ** 0.5), dim=-1)
        context = attn @ mem_proj
        return agent_state + torch.tanh(self.gate) * context

    def write_to_memory(self, agent_state: torch.Tensor,
                        episodic_slots: torch.Tensor,
                        slot_idx: int) -> torch.Tensor:
        written = self.agent_to_episodic(agent_state[:, -1])
        if written.dim() > 1:
            written = written.mean(dim=0)
        episodic_slots[0, slot_idx % self.episodic_slots] = written.detach()
        return episodic_slots


class AgentContextManager:
    def __init__(self, max_context: int = 8192, window_size: int = 4096):
        self.max_context = max_context
        self.window_size = window_size
        self.buffer = []

    def add(self, entry: dict):
        self.buffer.append(entry)
        if len(self.buffer) > self.max_context:
            self.buffer = self.buffer[-self.window_size:]

    def get_recent(self, n: int = 1024) -> list:
        return self.buffer[-n:]

    def get_task_summary(self) -> str:
        if not self.buffer:
            return ""
        actions = [s.get('action', {}) for s in self.buffer[-20:]]
        return f"Recent actions: {len(actions)} steps"


class ExperienceReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, state, action, reward, next_state, done):
        for name, t in [('state', state), ('action', action), ('next_state', next_state)]:
            if t is not None and (torch.isnan(t).any() or torch.isinf(t).any()):
                return
            if t is not None and t.numel() == 0:
                return
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size: int) -> tuple:
        if len(self.buffer) == 0:
            raise RuntimeError("Cannot sample from empty ExperienceReplayBuffer")
        if len(self.buffer) < batch_size:
            batch_size = len(self.buffer)
        indices = torch.randint(0, len(self.buffer), (batch_size,))
        batch = [self.buffer[i] for i in indices]
        states = torch.stack([b[0] for b in batch])
        actions = torch.stack([b[1] for b in batch])
        rewards = torch.tensor([b[2] for b in batch])
        next_states = torch.stack([b[3] for b in batch])
        dones = torch.tensor([b[4] for b in batch], dtype=torch.float)
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
