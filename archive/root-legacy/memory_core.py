import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import logging
logger = logging.getLogger(__name__)


class SurpriseGate(nn.Module):
    def __init__(self, d_model: int, d_mem: int):
        super().__init__()
        self.gate = nn.Linear(d_model, d_mem)
        self.lambda_proj = nn.Linear(d_model, 1)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        s = torch.sigmoid(self.gate(h))
        lam = torch.sigmoid(self.lambda_proj(h))
        return s, lam


class BiGRUSlotEncoder(nn.Module):
    def __init__(self, d_mem: int):
        super().__init__()
        assert d_mem % 2 == 0, "d_mem must be even for bidirectional GRU"
        self.gru = nn.GRU(d_mem, d_mem // 2, bidirectional=True, batch_first=True)
        self.proj = nn.Linear(d_mem, d_mem)

    def forward(self, slots: torch.Tensor) -> torch.Tensor:
        b, n, d = slots.shape
        if n == 0:
            return slots
        encoded, _ = self.gru(slots)
        encoded = self.proj(encoded)
        return encoded


class LTSMemory(nn.Module):
    def __init__(self, d_mem: int, capacity: int):
        super().__init__()
        self.d_mem = d_mem
        self.capacity = capacity
        self.mem = nn.Parameter(torch.zeros(1, capacity, d_mem))
        self.age = nn.Parameter(torch.zeros(1, capacity), requires_grad=False)
        nn.init.xavier_uniform_(self.mem)

    def write(self, keys: torch.Tensor, values: torch.Tensor, importance: torch.Tensor):
        b, k_orig, d = keys.shape
        scores = keys @ self.mem.transpose(-2, -1)
        usage = scores.softmax(dim=-1).sum(dim=1)
        combined = importance.mean(dim=[1, 2])
        write_priority = combined + 0.1 * usage
        k = min(k_orig, self.capacity)
        indices = write_priority.topk(k, dim=-1).indices
        values_write = values[:, :k] if k < k_orig else values
        scatter = torch.zeros(b, self.capacity, d, device=keys.device, dtype=keys.dtype)
        scatter.scatter_(1, indices.unsqueeze(-1).expand(-1, -1, d), values_write)
        MEM_UPDATE_MOMENTUM = 0.99
        MEM_NEW_DATA_RATE = 0.01
        with torch.no_grad():
            self.mem.data = self.mem.data * MEM_UPDATE_MOMENTUM + scatter.mean(dim=0, keepdim=True) * MEM_NEW_DATA_RATE

    def read(self, query: torch.Tensor) -> torch.Tensor:
        attn = query @ self.mem.transpose(-2, -1)
        attn = attn / math.sqrt(self.d_mem)
        attn = F.softmax(attn, dim=-1)
        out = attn @ self.mem
        return out


class GraphConsolidator(nn.Module):
    def __init__(self, d_mem: int, threshold: float = 0.65):
        super().__init__()
        self.threshold = threshold
        self.proj = nn.Linear(d_mem, d_mem)

    def forward(self, episodic_slots: torch.Tensor) -> torch.Tensor:
        b, n, d = episodic_slots.shape
        proj = self.proj(episodic_slots)
        proj = F.normalize(proj, dim=-1)
        sim = proj @ proj.transpose(-2, -1)
        adj = (sim > self.threshold).float()
        deg = adj.sum(dim=-1, keepdim=True).clamp(min=1)
        norm = adj / deg
        cluster_feats = norm @ episodic_slots
        return cluster_feats


class AurelianMemoryCore(nn.Module):
    def __init__(self, d_model: int, d_mem: int, episodic_slots: int, lts_capacity: int,
                 consolidation_freq: int = 64, graph_threshold: float = 0.65):
        super().__init__()
        self.d_model = d_model
        self.d_mem = d_mem
        self.consolidation_freq = consolidation_freq

        self.q_proj = nn.Linear(d_model, d_mem)
        self.k_proj = nn.Linear(d_model, d_mem)
        self.v_proj = nn.Linear(d_model, d_mem)
        self.out_proj = nn.Linear(d_mem, d_model)

        self.surprise = SurpriseGate(d_model, d_mem)
        self.episodic_encoder = BiGRUSlotEncoder(d_mem)
        self.lts = LTSMemory(d_mem, lts_capacity)
        self.graph = GraphConsolidator(d_mem, graph_threshold)

        self.forget_gate = nn.Linear(d_model, d_mem)
        self.gate_out = nn.Linear(d_model + d_mem, d_mem)

        self.episodic_slots_max = episodic_slots
        self.register_buffer('step_counter', torch.zeros(1, dtype=torch.long))

    def forward(self, h: torch.Tensor, return_mem_state: bool = False):
        b, t, d = h.shape
        self.step_counter += 1

        q = self.q_proj(h)
        k = self.k_proj(h)
        v = self.v_proj(h)
        surprise_scores, lambda_w = self.surprise(h)

        mem_read = self.lts.read(q)
        forget = torch.sigmoid(self.forget_gate(h))
        gated_mem = mem_read * forget

        gate_weights = torch.sigmoid(self.gate_out(torch.cat([h, gated_mem], dim=-1)))
        output = h + self.out_proj(gate_weights * gated_mem)

        if self.training and self.step_counter.item() % self.consolidation_freq == 0:
            self.lts.write(k, v, surprise_scores)

        mem_state = {
            'surprise': surprise_scores,
            'lambda': lambda_w,
            'mem_read': mem_read,
        }

        if return_mem_state:
            return output, mem_state
        return output
