import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class MoEMemoryRouter(nn.Module):
    def __init__(self, d_model, n_experts=8, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, n_experts, bias=False)

    def forward(self, x):
        logits = self.router(x)
        if self.training:
            noise = torch.randn_like(logits) * (1.0 / max(self.n_experts, 1))
            logits = logits + noise
        weights = F.softmax(logits, dim=-1)
        top_weights, top_indices = torch.topk(weights, self.top_k, dim=-1)
        top_weights = top_weights / (top_weights.sum(dim=-1, keepdim=True) + 1e-8)
        return top_weights, top_indices


class MemoryExpert(nn.Module):
    def __init__(self, d_mem, capacity):
        super().__init__()
        self.d_mem = d_mem
        self.capacity = capacity
        self.memory = nn.Parameter(torch.randn(capacity, d_mem) * 0.1)

    def read(self, query):
        attn = torch.matmul(query, self.memory.transpose(-2, -1)) / math.sqrt(self.d_mem)
        attn_weights = F.softmax(attn, dim=-1)
        return torch.matmul(attn_weights, self.memory)

    def write(self, keys, values):
        with torch.no_grad():
            scores = torch.matmul(keys, self.memory.transpose(-2, -1)) / math.sqrt(self.d_mem)
            write_weights = F.softmax(scores, dim=-1)
            update = torch.matmul(write_weights.transpose(-2, -1), values)
            self.memory.data = self.memory.data + update
            self.memory.data = torch.clamp(self.memory.data, -10.0, 10.0)

    def consolidate(self, lr=0.1):
        with torch.no_grad():
            self.memory.data = (1.0 - lr) * self.memory.data + lr * torch.tanh(self.memory.data)

    def forward(self, query):
        return self.read(query)


class MoELTSMemory(nn.Module):
    def __init__(self, d_model, d_mem, n_experts=8, top_k=2, capacity=1024):
        super().__init__()
        self.d_model = d_model
        self.d_mem = d_mem
        self.n_experts = n_experts
        self.top_k = top_k

        self.router = MoEMemoryRouter(d_model, n_experts, top_k)
        self.experts = nn.ModuleList([
            MemoryExpert(d_mem, capacity) for _ in range(n_experts)
        ])
        self.in_proj = nn.Linear(d_model, d_mem)
        self.out_proj = nn.Linear(d_mem, d_model)

    def forward(self, hidden_states, query):
        top_weights, top_indices = self.router(hidden_states)

        with torch.no_grad():
            expert_mask = F.one_hot(top_indices[..., 0], num_classes=self.n_experts).float()
            dispatch_fraction = expert_mask.mean(dim=(0, 1))

        all_probs = F.softmax(self.router.router(hidden_states), dim=-1)
        prob_fraction = all_probs.mean(dim=(0, 1))
        load_balancing_loss = self.n_experts * (dispatch_fraction * prob_fraction).sum()

        mem_query = self.in_proj(query)
        batch, seq, _ = mem_query.shape
        output = torch.zeros(batch, seq, self.d_mem, device=hidden_states.device)

        for e in range(self.n_experts):
            expert = self.experts[e]
            expert_out = expert.read(mem_query)

            for k in range(self.top_k):
                mask = top_indices[..., k] == e
                w = top_weights[..., k:k+1].expand(-1, -1, self.d_mem)
                output = output + w * expert_out * mask.unsqueeze(-1).float()

        output = self.out_proj(output)
        return output, load_balancing_loss
