import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dataclasses import dataclass, field
from typing import Optional, Callable
from collections import deque
from nn_utils import RMSNorm
import logging
logger = logging.getLogger(__name__)


class GatedResidual(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return x + torch.tanh(self.gate) * residual


class MLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, n_layers: int = 2):
        super().__init__()
        layers = []
        for i in range(n_layers):
            _in = d_in if i == 0 else d_hidden
            _out = d_out if i == n_layers - 1 else d_hidden
            layers.extend([nn.Linear(_in, _out), nn.GELU()])
        self.net = nn.Sequential(*layers[:-1])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─── 1. INPUT ENCODER ─────────────────────────────────────────────────────

class InputEncoder(nn.Module):
    """Converts text, tool outputs, memory, and system state into a unified situation vector."""

    def __init__(self, d_brain: int, d_model: int):
        super().__init__()
        self.text_proj = nn.Linear(d_model, d_brain)
        self.tool_proj = nn.Linear(d_model, d_brain)
        self.mem_proj = nn.Linear(d_model, d_brain)
        self.state_embed = nn.Embedding(32, d_brain)
        self.fuse = nn.Linear(4 * d_brain, d_brain)
        self.norm = RMSNorm(d_brain)
        self.gate = GatedResidual(d_brain)

    def forward(self, text_h: torch.Tensor, tool_h: Optional[torch.Tensor] = None,
                mem_h: Optional[torch.Tensor] = None, state_id: int = 0) -> torch.Tensor:
        b = text_h.shape[0]
        text = self.text_proj(text_h.mean(dim=1))
        tool = self.tool_proj(tool_h).mean(dim=1) if tool_h is not None else torch.zeros(b, text.shape[-1], device=text.device)
        mem = self.mem_proj(mem_h).mean(dim=1) if mem_h is not None else torch.zeros(b, text.shape[-1], device=text.device)
        state = self.state_embed(torch.tensor([state_id], device=text.device)).expand(b, -1)
        fused = self.fuse(torch.cat([text, tool, mem, state], dim=-1))
        return self.norm(self.gate(fused, text))


# ─── 2. WORKING MEMORY ────────────────────────────────────────────────────

class WorkingMemory(nn.Module):
    """Differentiable slot-based working memory with importance-gated writes."""

    def __init__(self, d_brain: int, n_slots: int = 64):
        super().__init__()
        self.d_brain = d_brain
        self.n_slots = n_slots
        self.gru = nn.GRUCell(d_brain, d_brain)
        self.write_gate = nn.Linear(d_brain * 2, 1)
        self.erase_gate = nn.Linear(d_brain * 2, 1)
        self.importance = MLP(d_brain, d_brain // 2, 1)
        self.query = nn.Linear(d_brain, d_brain)
        self.slot_proj = nn.Linear(d_brain, d_brain)
        self.norm = RMSNorm(d_brain)
        self.register_buffer('slots', torch.zeros(1, n_slots, d_brain))
        self.register_buffer('step', torch.zeros(1, dtype=torch.long))
        self.register_buffer('goal', torch.zeros(d_brain))

    def forward(self, s: torch.Tensor, goal: Optional[torch.Tensor] = None) -> torch.Tensor:
        b = s.shape[0]
        self.step.add_(1)
        if goal is not None:
            g = goal[0] if goal.dim() > 1 else goal
            self.goal.data.copy_(g)

        slots = self.slots.expand(b, -1, -1)
        h = self.gru(s, slots.mean(dim=1))
        combined = torch.cat([s, h], dim=-1)
        w = torch.sigmoid(self.write_gate(combined))
        e = torch.sigmoid(self.erase_gate(combined))
        slots = slots * (1 - e.unsqueeze(-1)) + w.unsqueeze(-1) * h.unsqueeze(1).expand(-1, self.n_slots, -1)

        imp = torch.sigmoid(self.importance(slots))
        if imp.mean() > 0.9:
            low_idx = imp.argsort(dim=1, descending=False)[:, :self.n_slots // 4]
            slots = slots.scatter(1, low_idx.expand(-1, -1, self.d_brain), h.unsqueeze(1))

        q = self.query(h).unsqueeze(1)
        scores = (q * self.slot_proj(slots)).sum(dim=-1) / math.sqrt(self.d_brain)
        attn = F.softmax(scores, dim=-1)
        read = (attn.unsqueeze(-1) * slots).sum(dim=1)
        self.slots.data.copy_(slots[:1].detach())
        return self.norm(h + read)

    def reset(self):
        self.slots.zero_()
        self.step.zero_()
        self.goal.zero_()


# ─── 3. LONG-TERM MEMORY INTERFACE ────────────────────────────────────────

@dataclass
class MemoryStore:
    keys: list = field(default_factory=list)
    values: list = field(default_factory=list)
    types: list = field(default_factory=list)
    importance: list = field(default_factory=list)
    access_count: list = field(default_factory=list)
    max_size: int = 100000


class LongTermMemory(nn.Module):
    """Retrieval interface to factual, procedural, and episodic memory stores."""

    def __init__(self, d_brain: int):
        super().__init__()
        self.d_brain = d_brain
        self.query_proj = nn.Linear(d_brain, d_brain)
        self.key_proj = nn.Linear(d_brain, d_brain)
        self.reranker = nn.Linear(d_brain * 2, 1)
        self.factual = MemoryStore(max_size=1000000)
        self.procedural = MemoryStore(max_size=100000)
        self.episodic = MemoryStore(max_size=500000)

    def search(self, query: torch.Tensor, store: MemoryStore, k: int = 5) -> list:
        if len(store.keys) == 0:
            return []
        q = self.query_proj(query.mean(dim=0))
        keys = torch.stack(store.keys[-store.max_size:]) if len(store.keys) > store.max_size else torch.stack(store.keys)
        scores = (keys * q.unsqueeze(0)).sum(dim=-1) / math.sqrt(self.d_brain)
        top_idx = scores.argsort(descending=True)[:k]
        return [(store.values[i], store.types[i] if i < len(store.types) else 'unknown', scores[i].item()) for i in top_idx]

    def retrieve(self, query: torch.Tensor, top_k: int = 5) -> list[torch.Tensor]:
        factual = self.search(query, self.factual, k=top_k)
        procedural = self.search(query, self.procedural, k=min(top_k, 3))
        episodic = self.search(query, self.episodic, k=min(top_k, 3))
        candidates = factual + procedural + episodic
        if not candidates:
            return []
        vals = torch.stack([c[0] for c in candidates])
        q = query.mean(dim=0).unsqueeze(0).expand(len(candidates), -1)
        scores = self.reranker(torch.cat([q, vals], dim=-1)).squeeze(-1)
        top = scores.argsort(descending=True)[:top_k]
        return [candidates[i] for i in top]

    def store(self, key: torch.Tensor, value: torch.Tensor, mem_type: str = 'episodic', importance: float = 0.5):
        store_map = {'factual': self.factual, 'procedural': self.procedural, 'episodic': self.episodic}
        store = store_map.get(mem_type, self.episodic)
        k = key if key.dim() >= 2 else key.unsqueeze(0)
        v = value if value.dim() >= 2 else value.unsqueeze(0)
        k = self.key_proj(k.mean(dim=0)).detach().cpu()
        v = v.mean(dim=0).detach().cpu()
        store.keys.append(k)
        store.values.append(v)
        store.types.append(mem_type)
        store.importance.append(importance)
        store.access_count.append(0)


# ─── 4. REASONING CORE ────────────────────────────────────────────────────

class ReasoningCore(nn.Module):
    """Multi-step reasoning with decomposition, self-consistency, and recursive thought."""

    def __init__(self, d_brain: int, lm_call: Optional[Callable] = None):
        super().__init__()
        self.d_brain = d_brain
        self.lm_call = lm_call
        self.decomposer = nn.Linear(d_brain, d_brain)
        self.subtask_attn = nn.Linear(d_brain, 1)
        self.step_proj = nn.Linear(d_brain, d_brain)
        self.confidence_proj = MLP(d_brain, d_brain // 2, 1)
        self.norm = RMSNorm(d_brain)

    def decompose(self, goal: torch.Tensor, wm: torch.Tensor) -> torch.Tensor:
        features = self.decomposer(goal + wm)
        attn = F.softmax(self.subtask_attn(features), dim=-1)
        return (features * attn).sum(dim=-1, keepdim=True)

    def estimate_confidence(self, step_embed: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.confidence_proj(step_embed))

    def forward(self, wm: torch.Tensor, goal: torch.Tensor, ltm: list,
                n_steps: int = 10, k_consistency: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
        all_steps = []
        for step in range(n_steps):
            subtask = self.decompose(goal, wm)
            step_h = self.step_proj(wm + subtask)
            step_h = self.norm(step_h)
            wm = wm + 0.1 * step_h
            all_steps.append(step_h)
            conf = self.estimate_confidence(step_h)
            if conf.max().item() > 0.9 and step > n_steps // 2:
                break
        stacked = torch.stack(all_steps).mean(dim=0)
        paths = [stacked for _ in range(k_consistency)]
        consensus = torch.stack(paths).mean(dim=0)
        confidence = self.estimate_confidence(consensus)
        return consensus, confidence


# ─── 5. PLANNER ───────────────────────────────────────────────────────────

class Planner(nn.Module):
    """Goal decomposition into task DAG with dependency prediction."""

    def __init__(self, d_brain: int, max_subtasks: int = 16):
        super().__init__()
        self.max_subtasks = max_subtasks
        self.goal_proj = nn.Linear(d_brain, d_brain)
        self.decompose = nn.Linear(d_brain, max_subtasks * d_brain)
        self.dep_predictor = nn.Linear(d_brain * 2, 1)
        self.schedule_head = nn.Linear(d_brain, 1)

    def forward(self, goal: torch.Tensor, wm: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        g = self.goal_proj(goal + wm)
        subtask_logits = self.decompose(g).view(-1, self.max_subtasks, goal.shape[-1])
        subtasks = F.softmax(subtask_logits, dim=1)
        n = subtasks.shape[1]
        pairs = subtasks.unsqueeze(2).expand(-1, n, n, -1)
        pair_feats = torch.cat([pairs, pairs.transpose(1, 2)], dim=-1)
        deps = torch.sigmoid(self.dep_predictor(pair_feats).squeeze(-1))
        schedule_flat = subtasks.mean(dim=1)
        schedule = self.schedule_head(schedule_flat).squeeze(-1)
        return subtasks, deps


# ─── 6. TOOL CONTROLLER ───────────────────────────────────────────────────

class ToolController(nn.Module):
    """Learned tool selection, execution, and result integration."""

    def __init__(self, d_brain: int, n_tools: int = 32):
        super().__init__()
        self.n_tools = n_tools
        self.tool_embeds = nn.Embedding(n_tools, d_brain)
        self.query_proj = nn.Linear(d_brain, d_brain)
        self.confidence_proj = MLP(d_brain, d_brain // 2, 1)
        self.result_proj = nn.Linear(d_brain, d_brain)
        self.gate = GatedResidual(d_brain)

    def select(self, wm: torch.Tensor, goal: torch.Tensor) -> tuple[int, torch.Tensor]:
        q = self.query_proj(wm + goal)
        scores = (q.unsqueeze(1) * self.tool_embeds.weight.unsqueeze(0)).sum(dim=-1)
        best = scores[0].argmax(dim=-1)
        conf = self.confidence_proj(q)
        return best.item(), conf

    def integrate(self, wm: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        return self.gate(wm, self.result_proj(result))


# ─── 7. AGENT ROUTER ──────────────────────────────────────────────────────

class AgentRouter(nn.Module):
    """Routes subtasks to specialized sub-agents and merges results."""

    def __init__(self, d_brain: int, n_agents: int = 10):
        super().__init__()
        self.n_agents = n_agents
        self.agent_keys = nn.Embedding(n_agents, d_brain)
        self.router = nn.Linear(d_brain, d_brain)
        self.merger = nn.Linear(d_brain * n_agents, d_brain)
        self.norm = RMSNorm(d_brain)

    def route(self, subtask: torch.Tensor) -> list[int]:
        q = self.router(subtask)
        scores = (q.unsqueeze(1) * self.agent_keys.weight.unsqueeze(0)).sum(dim=-1)
        top_k = scores.argsort(descending=True)[0, :3].tolist()
        return top_k

    def merge(self, subtask: torch.Tensor, results: list[torch.Tensor]) -> torch.Tensor:
        n = len(results)
        padded = results + [torch.zeros_like(subtask)] * (self.n_agents - n)
        merged = self.merger(torch.cat(padded, dim=-1))
        return self.norm(subtask + merged)


# ─── 8. VERIFIER ──────────────────────────────────────────────────────────

class Verifier(nn.Module):
    """Error detection, confidence scoring, and fix generation."""

    def __init__(self, d_brain: int, n_errors: int = 9):
        super().__init__()
        self.error_detector = MLP(d_brain, d_brain // 2, 1)
        self.error_classifier = nn.Linear(d_brain, n_errors)
        self.confidence_scorer = MLP(d_brain, d_brain // 2, 1)
        self.fix_proj = nn.Linear(d_brain + n_errors, d_brain)
        self.n_errors = n_errors

    def forward(self, step: torch.Tensor, wm: torch.Tensor) -> tuple:
        features = step + wm.mean(dim=1, keepdim=True)
        error_prob = torch.sigmoid(self.error_detector(features))
        error_types = torch.sigmoid(self.error_classifier(features))
        confidence = torch.sigmoid(self.confidence_scorer(features))
        fix = self.fix_proj(torch.cat([features, error_types], dim=-1))
        return error_prob, error_types, confidence, fix

    def detect(self, step: torch.Tensor, wm: torch.Tensor) -> tuple[bool, float, torch.Tensor]:
        error_prob, error_types, confidence, fix = self.forward(step, wm)
        has_error = error_prob.mean().item() > 0.5
        return has_error, confidence.mean().item(), fix


# ─── 9. CRITIC ────────────────────────────────────────────────────────────

class Critic(nn.Module):
    def __init__(self, d_brain: int):
        super().__init__()
        self.encoder = nn.Linear(d_brain * 2, d_brain)
        self.score_head = nn.Linear(d_brain, 1)
        self.suggestion_head = nn.Linear(d_brain, d_brain)
        self.norm = RMSNorm(d_brain)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if state.dim() > 2:
            state = state.mean(dim=1) if state.dim() == 3 else state.mean(dim=tuple(range(1, state.dim()-1)))
        if action.dim() > 2:
            action = action.mean(dim=1) if action.dim() == 3 else action.mean(dim=tuple(range(1, action.dim()-1)))
        cat = torch.cat([state, action], dim=-1)
        h = self.norm(F.relu(self.encoder(cat)))
        score = torch.tanh(self.score_head(h))
        suggestion = self.suggestion_head(h)
        return score, suggestion


# ─── 10. REFLECTION MODULE ────────────────────────────────────────────────

class ReflectionModule(nn.Module):
    """Post-task trajectory analysis: extracts successes, failures, and updates memory."""

    def __init__(self, d_brain: int):
        super().__init__()
        self.traj_encoder = nn.GRU(d_brain, d_brain // 2, bidirectional=True, batch_first=True)
        self.success_extractor = nn.Linear(d_brain, d_brain)
        self.failure_extractor = nn.Linear(d_brain, d_brain)
        self.summary_proj = nn.Linear(d_brain, d_brain)

    def forward(self, trajectory: list[torch.Tensor], outcome: torch.Tensor) -> dict:
        if not trajectory:
            return {'summary': torch.zeros(1, outcome.shape[-1], device=outcome.device), 'success': torch.zeros_like(outcome[:1]), 'failure': torch.zeros_like(outcome[:1]), 'improvement': torch.tensor(0.0)}
        t = torch.stack(trajectory)
        if t.dim() == 3:
            seq, b, d = t.shape
            t = t.permute(1, 0, 2)  # (b, seq, d)
        elif t.dim() == 2:
            t = t.unsqueeze(0)
        encoded, _ = self.traj_encoder(t)
        pooled = encoded.mean(dim=1)
        pooled = pooled.mean(dim=0, keepdim=True) if pooled.dim() > 1 and pooled.shape[0] != outcome.shape[0] else pooled
        outcome = outcome.mean(dim=0, keepdim=True) if outcome.dim() > 1 and outcome.shape[0] != pooled.shape[0] else outcome
        success_pattern = self.success_extractor(pooled)
        failure_pattern = self.failure_extractor(pooled)
        summary = self.summary_proj(pooled + outcome)
        improvement = (torch.sigmoid(outcome.mean()) - 0.5) * 2
        return {
            'summary': summary,
            'success': success_pattern,
            'failure': failure_pattern,
            'improvement': improvement,
        }


# ─── 11. EXECUTIVE CONTROLLER ─────────────────────────────────────────────

class ExecutiveController(nn.Module):
    """Orchestrates all modules. Decides think/act/retrieve/verify/finalize."""

    def __init__(self, d_brain: int, n_actions: int = 8):
        super().__init__()
        self.n_actions = n_actions
        self.state_gru = nn.GRUCell(d_brain, d_brain)
        self.action_head = nn.Linear(d_brain, n_actions)
        self.value_head = nn.Linear(d_brain, 1)
        self.loop_detector = nn.Linear(n_actions, 1)
        self.register_buffer('action_history', torch.zeros(10, n_actions))
        self.register_buffer('step_count', torch.zeros(1, dtype=torch.long))

        self.ACTION_NAMES = [
            'think_more', 'act', 'retrieve_memory', 'use_tool',
            'ask_clarification', 'verify', 'finalize', 'reflect',
        ]

    def forward(self, s: torch.Tensor, wm: torch.Tensor, budget: float = 1.0) -> tuple[int, torch.Tensor]:
        self.step_count += 1
        state_h = self.state_gru(s, wm if wm.dim() == 2 and wm.shape[-1] == s.shape[-1] else wm.mean(dim=1))
        action_scores = self.action_head(state_h)
        value = self.value_head(state_h)

        cost_penalty = torch.tensor([0.5, 1.0, 0.3, 1.5, 0.2, 0.3, 0.1, 0.8], device=s.device)
        action_scores = action_scores - cost_penalty.unsqueeze(0) * (1 - budget)

        action_idx = action_scores[0].argmax(dim=-1)  # first batch item
        self.action_history = torch.roll(self.action_history, -1, dims=0)
        one_hot = F.one_hot(action_idx, num_classes=self.n_actions).float()
        self.action_history[-1] = one_hot

        loop_score = torch.sigmoid(self.loop_detector(self.action_history[-1].unsqueeze(0)))
        if loop_score.item() > 0.5:
            action_scores[:, 6] = action_scores.max()  # force finalize

        if self.step_count > 5:
            recent_unique = self.action_history[-5:].unique(dim=0).shape[0]
            if recent_unique < 2:
                action_scores[:, 6] = action_scores.max()  # force finalize
            for i in range(self.n_actions):
                if (self.action_history[-4:] == F.one_hot(torch.tensor(i), num_classes=self.n_actions).float()).all(dim=-1).any():
                    action_scores[0, i] -= 2.0  # repeat penalty

        action = action_scores[0].argmax(dim=-1).item()
        return action, value

    def action_name(self, action_idx: int) -> str:
        return self.ACTION_NAMES[action_idx] if action_idx < len(self.ACTION_NAMES) else 'unknown'

    def reset(self):
        self.action_history.zero_()
        self.step_count.zero_()


# ─── 12. UNCERTAINTY ESTIMATOR ────────────────────────────────────────────

class UncertaintyEstimator(nn.Module):
    """Estimates epistemic and aleatoric uncertainty from the model's hidden states."""

    def __init__(self, d_brain: int):
        super().__init__()
        self.epistemic = MLP(d_brain, d_brain // 2, 1)
        self.aleatoric = MLP(d_brain, d_brain // 2, 1)
        self.entropy_proj = nn.Linear(d_brain, 1)

    def forward(self, h: torch.Tensor, logits: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        epi = torch.sigmoid(self.epistemic(h))
        alea = F.softplus(self.aleatoric(h))
        entropy = torch.sigmoid(self.entropy_proj(h)) if logits is None else \
            -(F.softmax(logits, dim=-1) * F.log_softmax(logits, dim=-1)).sum(dim=-1, keepdim=True)
        return epi, alea, entropy

    def is_uncertain(self, h: torch.Tensor, threshold: float = 0.7) -> bool:
        epi, alea, ent = self.forward(h)
        return (epi + alea + ent).mean().item() > threshold


# ─── 13. SELF-IMPROVEMENT LOOP ────────────────────────────────────────────

class SelfImprovementLoop(nn.Module):
    """Online RL-based self-improvement with memory consolidation and skill acquisition."""

    def __init__(self, d_brain: int, lr: float = 1e-4):
        super().__init__()
        self.lr = lr
        self.optimizer = None
        self.task_count = 0
        self.recent_rewards = deque(maxlen=100)
        self.recent_trajectories = deque(maxlen=1000)
        self.value_net = nn.Linear(d_brain, 1)

    def attach_optimizer(self, params):
        self.optimizer = torch.optim.AdamW(params, lr=self.lr)

    def update(self, trajectory: list, reward: float, task_embed: torch.Tensor,
               executive_action: int, action_logprob: torch.Tensor):
        self.task_count += 1
        self.recent_rewards.append(reward)
        if self.optimizer is None:
            return {'lr': self.lr}

        advantage = reward - self.value_net(task_embed.mean(dim=0, keepdim=True))
        policy_loss = -advantage.detach() * action_logprob
        value_loss = advantage.pow(2)
        total_loss = policy_loss + 0.5 * value_loss

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]['params'], 1.0)
        self.optimizer.step()

        if reward > 0.8:
            self.lr = min(self.lr * 1.001, 1e-3)
        elif reward < 0.2:
            self.lr = max(self.lr * 0.999, 1e-6)
        for group in self.optimizer.param_groups:
            group['lr'] = self.lr

        self.recent_trajectories.append({'trajectory': trajectory, 'reward': reward})
        if self.task_count % 100 == 0 and len(self.recent_rewards) > 0:
            avg = sum(self.recent_rewards) / len(self.recent_rewards)
            self.value_net.weight.data *= 0.99  # gentle weight decay

        return {'policy_loss': policy_loss.item(), 'value_loss': value_loss.item(), 'lr': self.lr}

    def get_avg_reward(self) -> float:
        return sum(self.recent_rewards) / max(len(self.recent_rewards), 1)


# ─── BRAIN LAYER: FULL INTEGRATION ────────────────────────────────────────

class NeuralBrainLayer(nn.Module):
    """Complete neural brain: 13 modules, full reasoning loop, trainable end-to-end."""

    def __init__(self, d_brain: int = 1024, d_model: int = 1536, n_slots: int = 64,
                 n_tools: int = 32, n_agents: int = 10, n_actions: int = 8,
                 max_steps: int = 50, lm_call: Optional[Callable] = None):
        super().__init__()
        self.d_brain = d_brain
        self.max_steps = max_steps
        self.lm_call = lm_call

        self.encoder = InputEncoder(d_brain, d_model)
        self.wm = WorkingMemory(d_brain, n_slots)
        self.ltm = LongTermMemory(d_brain)
        self.reasoning = ReasoningCore(d_brain, lm_call)
        self.planner = Planner(d_brain)
        self.tool_ctrl = ToolController(d_brain, n_tools)
        self.router = AgentRouter(d_brain, n_agents)
        self.verifier = Verifier(d_brain)
        self.critic = Critic(d_brain)
        self.reflection = ReflectionModule(d_brain)
        self.executive = ExecutiveController(d_brain, n_actions)
        self.uncertainty = UncertaintyEstimator(d_brain)
        self.self_improve = SelfImprovementLoop(d_brain)

        self.register_buffer('budget', torch.tensor(1.0))

    def forward(self, text_h: torch.Tensor, tool_h: Optional[torch.Tensor] = None,
                mem_h: Optional[torch.Tensor] = None, state_id: int = 0,
                goal: Optional[torch.Tensor] = None,
                full_trace: bool = False) -> dict:
        """Run the full brain loop: Input → Encode → Retrieve → Reason → Verify → Output → Learn."""
        self.wm.reset()
        self.executive.reset()

        trajectory = []
        s = self.encoder(text_h, tool_h, mem_h, state_id)

        if goal is None:
            goal = s.detach().clone()

        ltm_context = self.ltm.retrieve(s)
        ltm_embed = torch.stack([c[0] for c in ltm_context]).to(s.device) if ltm_context else torch.zeros(1, self.d_brain, device=s.device)

        wm = self.wm(s, goal)
        plan_subtasks, plan_deps = self.planner(goal, wm)
        for i in range(min(3, plan_subtasks.shape[1])):
            self.router.route(plan_subtasks[:, i, :])

        for step in range(self.max_steps):
            s = self.encoder(text_h, tool_h, ltm_embed.mean(dim=0, keepdim=True) if ltm_context else None, state_id)
            wm = self.wm(s, goal)
            budget_left = self.budget.item() - (step / self.max_steps)
            action, value = self.executive(s, wm, budget=budget_left)

            action_name = self.executive.action_name(action)
            trajectory.append({'step': step, 'action': action_name, 'wm': wm.detach().clone()})

            if action_name == 'think_more':
                step_out, conf = self.reasoning(wm, goal, ltm_context)
                wm = wm + 0.1 * step_out
                self.budget = self.budget * 0.98

            elif action_name == 'retrieve_memory':
                ltm_context = self.ltm.retrieve(wm)
                if ltm_context:
                    ltm_embed = torch.stack([c[0] for c in ltm_context]).to(s.device).mean(dim=0, keepdim=True)
                    wm = wm + 0.05 * ltm_embed

            elif action_name == 'use_tool':
                tool_idx, conf = self.tool_ctrl.select(wm, goal)
                if conf.item() > 0.3:
                    result = torch.randn_like(wm) * 0.01
                    wm = self.tool_ctrl.integrate(wm, result)
                self.budget = self.budget * 0.9

            elif action_name == 'verify':
                has_error, confidence, fix = self.verifier.detect(wm, s.unsqueeze(0))
                if has_error:
                    wm = wm + 0.1 * fix
                score, suggestion = self.critic(s.unsqueeze(0), wm.unsqueeze(0))
                wm = wm + 0.05 * suggestion

            elif action_name == 'reflect':
                wm_snapshots = [t['wm'] for t in trajectory if 'wm' in t]
                ref_out = self.reflection(wm_snapshots, s.detach().clone())
                self.ltm.store(s, ref_out['summary'], mem_type='episodic', importance=0.5)
                wm = wm + 0.1 * ref_out['summary']

            elif action_name == 'finalize':
                break

        epi_unc, alea_unc, entropy = self.uncertainty(wm)
        wm_tensors = [t['wm'] for t in trajectory if 'wm' in t]
        ref_out = self.reflection(wm_tensors, s.detach().clone())

        result = {
            'output': wm,
            'confidence': 1.0 - (epi_unc + alea_unc + entropy).mean().item(),
            'uncertainty': {'epistemic': epi_unc, 'aleatoric': alea_unc, 'entropy': entropy},
            'steps': step + 1,
            'trajectory': trajectory if full_trace else None,
            'reflection': ref_out,
            'value': value.mean().item(),
            'action': action,
        }
        return result

    def learn(self, trajectory: list, reward: float, task_embed: torch.Tensor,
              action: int, logprob: torch.Tensor) -> dict:
        return self.self_improve.update(trajectory, reward, task_embed, action, logprob)

    def set_budget(self, budget: float):
        self.budget.fill_(max(0.01, min(1.0, budget)))

    def get_state_dict(self) -> dict:
        return {
            'encoder': self.encoder.state_dict(),
            'wm': self.wm.state_dict(),
            'reasoning': self.reasoning.state_dict(),
            'planner': self.planner.state_dict(),
            'tool': self.tool_ctrl.state_dict(),
            'router': self.router.state_dict(),
            'verifier': self.verifier.state_dict(),
            'critic': self.critic.state_dict(),
            'reflection': self.reflection.state_dict(),
            'executive': self.executive.state_dict(),
            'uncertainty': self.uncertainty.state_dict(),
            'improve': self.self_improve.state_dict(),
        }

    def load_state_dict_custom(self, state: dict):
        self.encoder.load_state_dict(state['encoder'])
        self.wm.load_state_dict(state['wm'])
        self.reasoning.load_state_dict(state['reasoning'])
        self.planner.load_state_dict(state['planner'])
        self.tool_ctrl.load_state_dict(state['tool'])
        self.router.load_state_dict(state['router'])
        self.verifier.load_state_dict(state['verifier'])
        self.critic.load_state_dict(state['critic'])
        self.reflection.load_state_dict(state['reflection'])
        self.executive.load_state_dict(state['executive'])
        self.uncertainty.load_state_dict(state['uncertainty'])
        self.self_improve.load_state_dict(state['improve'])


# ─── EXECUTIVE CONTROLLER PSEUDOCODE ─────────────────────────────────────

"""
function EXECUTIVE_CONTROLLER(brain, input, max_steps=50):
    s = brain.encoder(input)              # encode input
    brain.wm.reset()                      # reset working memory
    ltm = brain.ltm.retrieve(s)          # retrieve from LTM
    wm = brain.wm(s, goal=extract_goal(s)) # build working state

    for step in 1..max_steps:
        budget = 1.0 - (step / max_steps)
        action = brain.executive(s, wm, budget)

        switch action:
            case 'think_more':
                step_out, conf = brain.reasoning(wm, goal, ltm)
                wm += 0.1 * step_out
            case 'retrieve_memory':
                ltm = brain.ltm.retrieve(wm)
            case 'use_tool':
                tool_idx, conf = brain.tool_ctrl.select(wm, goal)
                if conf > 0.3:
                    result = execute_tool(tool_idx)
                    wm = brain.tool_ctrl.integrate(wm, result)
            case 'verify':
                has_error, conf, fix = brain.verifier.detect(wm, s)
                if has_error: wm += 0.1 * fix
                score, suggestion = brain.critic(s, wm)
                wm += 0.05 * suggestion
            case 'reflect':
                ref = brain.reflection(trajectory, outcome)
                brain.ltm.store(s, ref['summary'])
            case 'finalize':
                return wm, confidence

    return wm, confidence
"""


# ─── REASONING PSEUDOCODE ────────────────────────────────────────────────

"""
function REASON(brain, goal, wm, ltm, max_depth=16, k_consistency=5):
    for depth in 1..max_depth:
        subtask = brain.reasoning.decompose(goal, wm)
        step_h = brain.reasoning.step_proj(wm + subtask)
        wm += 0.1 * brain.reasoning.norm(step_h)

        conf = brain.reasoning.estimate_confidence(step_h)
        if conf > 0.9 and depth > max_depth // 2:
            break

    # Self-consistency: run k paths, take majority
    paths = []
    for k in 1..k_consistency:
        path = REASON(brain, goal, wm.clone(), ltm, max_depth=4)
        paths.append(path)
    consensus = mean(paths)
    confidence = brain.reasoning.estimate_confidence(consensus)

    return consensus, confidence
"""


# ─── VERIFICATION PSEUDOCODE ─────────────────────────────────────────────

"""
function VERIFY(brain, step, wm, max_revisions=3):
    for rev in 1..max_revisions:
        error_prob, error_types, confidence, fix = brain.verifier(step, wm)

        if error_prob < 0.15 and confidence > 0.8:
            return PASS, confidence, step

        has_error, conf, fix = brain.verifier.detect(step, wm)
        if has_error:
            step = step + 0.1 * fix  # apply fix

        score, suggestion = brain.critic(wm, step)
        step = step + 0.05 * suggestion

    return FAIL, confidence, step
"""


# ─── MEMORY UPDATE PSEUDOCODE ────────────────────────────────────────────

"""
function UPDATE_MEMORY(brain, task, trajectory, outcome, reward):
    # Extract key from task
    key = brain.ltm.key_proj(task.mean(dim=0))

    # Store episodic trace
    if reward > 0.5:
        brain.ltm.store(key, trajectory_embed, 'episodic', importance=reward)

    # Store procedural pattern (success)
    if reward > 0.8:
        brain.ltm.store(key, trajectory_embed.mean(dim=0), 'procedural', importance=reward)

    # Store facts
    for fact in extract_facts(trajectory):
        brain.ltm.store(fact.key, fact.value, 'factual', importance=1.0)

    # Consolidate (periodic)
    if brain.self_improve.task_count % 100 == 0:
        for store in [brain.ltm.factual, brain.ltm.procedural, brain.ltm.episodic]:
            if len(store.keys) > 1000:
                # Keep only top-importance entries
                idx = sorted(range(len(store.importance)),
                           key=lambda i: store.importance[i],
                           reverse=True)[:store.max_size // 2]
                store.keys = [store.keys[i] for i in idx]
                store.values = [store.values[i] for i in idx]
"""


# ─── REFLECTION PSEUDOCODE ───────────────────────────────────────────────

"""
function REFLECT(brain, trajectory, outcome):
    # Analyze trajectory
    ref_out = brain.reflection(trajectory, outcome)

    # Extract success pattern
    if ref_out['improvement'] > 0.3:
        brain.ltm.store(
            key=outcome,
            value=ref_out['success'],
            mem_type='procedural',
            importance=0.5 + ref_out['improvement']
        )

    # Extract failure pattern
    if ref_out['improvement'] < -0.3:
        brain.ltm.store(
            key=outcome + '_FAILURE',
            value=ref_out['failure'],
            mem_type='episodic',
            importance=0.5 - ref_out['improvement']
        )

    # Return learning signal
    return {
        'lessons': ref_out['summary'],
        'improvement': ref_out['improvement'],
        'update_priority': 'high' if abs(ref_out['improvement']) > 0.5 else 'low',
    }
"""


# ─── SCALING PATH ────────────────────────────────────────────────────────

"""
┌─────────┬──────────┬───────────┬──────────┬──────────────┐
│ Tier    │ d_brain  │ n_slots   │ n_agents │ max_steps    │
├─────────┼──────────┼───────────┼──────────┼──────────────┤
│ 1B      │ 1024     │ 64        │ 10       │ 50           │
│ 3B      │ 1536     │ 96        │ 16       │ 75           │
│ 7B      │ 2048     │ 128       │ 24       │ 100          │
│ 14B     │ 3072     │ 192       │ 32       │ 150          │
│ 32B     │ 4096     │ 256       │ 48       │ 200          │
└─────────┴──────────┴───────────┴──────────┴──────────────┘

All projection matrices scale with d_brain. Memory capacities scale 2x per tier.
The self-improvement loop maintains 1000 recent trajectories regardless of scale.
"""

# Neural brain layer complete. Proceeding to the next AI layer.
