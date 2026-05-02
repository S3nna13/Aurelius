import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
logger = logging.getLogger(__name__)


class ToolEmbedding(nn.Module):
    def __init__(self, d_model: int, max_tools: int = 128, tool_desc_dim: int = 64):
        super().__init__()
        self.d_model = d_model
        self.tool_embed = nn.Embedding(max_tools, d_model)
        self.desc_proj = nn.Linear(tool_desc_dim, d_model)
        self.type_head = nn.Linear(d_model, max_tools)
        self.param_head = nn.Linear(d_model, d_model)

    def forward(self, tool_ids: torch.Tensor | None = None,
                tool_descs: torch.Tensor | None = None) -> torch.Tensor:
        if tool_ids is not None:
            return self.tool_embed(tool_ids)
        if tool_descs is not None:
            return self.desc_proj(tool_descs)
        return self.tool_embed.weight.mean(dim=0, keepdim=True)


class ToolCrossAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.q = nn.Linear(d_model, d_model, bias=False)
        self.k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, h: torch.Tensor, tool_embeds: torch.Tensor) -> torch.Tensor:
        b, t, d = h.shape
        q = self.q(h).view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k(tool_embeds).view(tool_embeds.shape[0], -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v(tool_embeds).view(tool_embeds.shape[0], -1, self.n_heads, self.head_dim).transpose(1, 2)

        attn = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, t, d)
        out = self.out(out)
        return h + torch.tanh(self.gate) * out


class ToolCallHead(nn.Module):
    def __init__(self, d_model: int, n_known_tools: int = 64, max_params: int = 8):
        super().__init__()
        self.tool_selector = nn.Linear(d_model, n_known_tools)
        self.param_presence = nn.Linear(d_model, max_params)
        self.param_values = nn.Linear(d_model, max_params * d_model // 8)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        tool_logits = self.tool_selector(h[:, -1])
        param_presence = torch.sigmoid(self.param_presence(h[:, -1]))
        param_raw = self.param_values(h[:, -1])
        return tool_logits, param_presence, param_raw


class ToolResultIntegrator(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.result_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, h: torch.Tensor, result: torch.Tensor) -> torch.Tensor:
        proj = self.result_proj(result)
        return h + torch.tanh(self.gate) * proj


class ToolFormerAdapter(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_known_tools: int = 64):
        super().__init__()
        self.tool_embed = ToolEmbedding(d_model, max_tools=n_known_tools)
        self.cross_attn = ToolCrossAttention(d_model, n_heads)
        self.call_head = ToolCallHead(d_model, n_known_tools=n_known_tools)
        self.result_integrator = ToolResultIntegrator(d_model)

    def forward(self, h: torch.Tensor, tool_descs: torch.Tensor | None = None,
                return_call: bool = False) -> tuple:
        tool_embeds = self.tool_embed() if tool_descs is None else self.tool_embed(tool_descs=tool_descs)
        b_tools = tool_embeds.shape[0]
        h_attended = self.cross_attn(h, tool_embeds.unsqueeze(0) if b_tools == 1 else tool_embeds)
        if return_call:
            tool_logits, params_presence, params_raw = self.call_head(h_attended)
            return h_attended, (tool_logits, params_presence, params_raw)
        return h_attended, None


class ValueHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
        )

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        if h.dim() == 3:
            h = h[:, -1]
        return self.proj(h).squeeze(-1)


class CriticHead(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
        )
        self.score = nn.Linear(d_model // 2, 1)
        self.suggestion = nn.Linear(d_model // 2, d_model)

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if state.dim() == 3:
            state = state[:, -1]
        if action.dim() == 3:
            action = action[:, -1]
        cat = torch.cat([state, action], dim=-1)
        encoded = self.encoder(cat)
        score = self.score(encoded).squeeze(-1)
        suggestion = self.suggestion(encoded)
        return score, suggestion


class MCTSNode:
    def __init__(self, state_embed: torch.Tensor, parent=None, action_idx: int = -1):
        self.state = state_embed
        self.parent = parent
        self.action_idx = action_idx
        self.children = []
        self.visits = 0
        self.value_sum = 0.0
        self.prior = 0.0

    def value(self) -> float:
        return self.value_sum / max(self.visits, 1)

    def ucb_score(self, c_puct: float = 1.4) -> float:
        if self.visits == 0:
            return float('inf')
        return self.value() + c_puct * self.prior * math.sqrt(max(self.parent.visits, 1)) / (1 + self.visits)


class PlanningModule(nn.Module):
    def __init__(self, d_model: int, n_simulations: int = 16, max_depth: int = 8):
        super().__init__()
        self.d_model = d_model
        self.n_simulations = n_simulations
        self.max_depth = max_depth
        self.value_head = ValueHead(d_model)
        self.action_proposer = nn.Linear(d_model, d_model)

    def forward(self, h: torch.Tensor, n_actions: int = 8) -> tuple[torch.Tensor, torch.Tensor, list]:
        b, t, d = h.shape
        root_state = h[:, -1]
        plans = []
        value_tensors = []

        effective_batch = min(b, 8)
        for batch_idx in range(effective_batch):
            root = MCTSNode(root_state[batch_idx:batch_idx+1].detach())
            for _ in range(self.n_simulations):
                node = root
                depth = 0
                while node.children and depth < self.max_depth:
                    node = max(node.children, key=lambda c: c.ucb_score())
                    depth += 1

                if depth < self.max_depth:
                    action_embeds = self.action_proposer(node.state)
                    expanded = action_embeds + node.state
                    child = MCTSNode(expanded.detach(), parent=node, action_idx=depth)
                    node.children.append(child)
                    v = self.value_head(expanded)
                    node.visits += 1
                    node.value_sum += v.item()
                else:
                    v = self.value_head(node.state)
                    node.visits += 1
                    node.value_sum += v.item()

                n = node
                while n.parent is not None:
                    n.parent.visits += 1
                    n.parent.value_sum += v.item() * 0.1
                    n = n.parent

            best = max(root.children, key=lambda c: c.visits) if root.children else root
            plans.append(best.state)
            value_tensors.append(self.value_head(root_state[batch_idx:batch_idx+1]).squeeze(-1))

        plan_tensor = torch.cat(plans, dim=0) if plans else root_state
        value_tensor = torch.stack([v.flatten() for v in value_tensors]).squeeze(-1) if value_tensors else torch.zeros(b, device=h.device)
        return plan_tensor, value_tensor, root
