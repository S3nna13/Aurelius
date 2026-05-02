import torch
import torch.nn as nn


class PredictiveMemoryPrefetcher:
    def __init__(self, d_model: int, d_mem: int, n_layers: int, history_len: int = 64):
        self.d_model = d_model
        self.d_mem = d_mem
        self.n_layers = n_layers
        self.history_len = history_len
        self.surprise_history = [[] for _ in range(n_layers)]
        self.prefetch_buffer = {}

    def observe(self, layer: int, surprise: torch.Tensor, hidden: torch.Tensor):
        self.surprise_history[layer].append(surprise.mean().item())
        if len(self.surprise_history[layer]) > self.history_len:
            self.surprise_history[layer].pop(0)

    def predict_next_surprise(self, layer: int) -> float:
        hist = self.surprise_history[layer]
        if len(hist) < 4:
            return 0.5
        recent = hist[-4:]
        slope = (recent[-1] - recent[0]) / max(len(recent), 1)
        return max(0.0, min(1.0, recent[-1] + slope))

    def should_prefetch(self, layer: int) -> bool:
        pred = self.predict_next_surprise(layer)
        return pred > 0.6

    def get_prefetch_indices(self, lts: torch.Tensor, query: torch.Tensor,
                              top_k: int = 32) -> torch.Tensor:
        scores = query @ lts.transpose(-2, -1)
        return scores.topk(top_k, dim=-1).indices

    def reset(self):
        self.surprise_history = [[] for _ in range(self.n_layers)]


class SparseLTSRouter(nn.Module):
    def __init__(self, d_model: int, d_mem: int, top_k: int = 32,
                 n_experts: int = 4, routing_noise: float = 0.1):
        super().__init__()
        self.top_k = top_k
        self.n_experts = n_experts
        self.router = nn.Linear(d_model, n_experts)
        self.noise_std = routing_noise
        self.expert_offsets = nn.Parameter(torch.zeros(n_experts), requires_grad=False)

    def forward(self, h: torch.Tensor, lts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = h.shape
        logits = self.router(h)
        if self.training and self.noise_std > 0:
            noise = torch.randn_like(logits) * self.noise_std
            logits = logits + noise
        expert_weights = torch.softmax(logits, dim=-1)
        expert_idx = expert_weights.argmax(dim=-1)
        n_lts = lts.shape[1]
        entries_per_expert = n_lts // self.n_experts
        offsets = expert_idx * entries_per_expert
        offsets = offsets.clamp(max=n_lts - self.top_k)
        n_lts = lts.shape[1]
        d = lts.shape[-1]
        indices = offsets.unsqueeze(-1) + torch.arange(self.top_k, device=lts.device)
        indices = indices.clamp(max=n_lts - 1)
        lts_exp = lts.expand(b, -1, -1).unsqueeze(1)
        indices_exp = indices.unsqueeze(-1).expand(-1, -1, -1, d)
        selected = torch.gather(lts_exp.expand(-1, t, -1, -1), 2, indices_exp)
        scores = h.unsqueeze(2) @ selected.transpose(-2, -1)
        attn = torch.softmax(scores / (d ** 0.5), dim=-1)
        result = (attn @ selected).squeeze(2)
        return result, expert_weights


class PredictiveAttentionRouter(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_memory_slots: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.mem_proj = nn.Linear(d_model, d_model)
        self.gate = nn.Linear(d_model, n_heads)

    def forward(self, h: torch.Tensor, memory: torch.Tensor) -> torch.Tensor:
        b, t, d = h.shape
        mem_proj = self.mem_proj(memory)
        h_proj = h.view(b, t, self.n_heads, self.head_dim).transpose(1, 2)
        mem_proj = mem_proj.view(1, -1, self.n_heads, self.head_dim).transpose(1, 2)
        attn = torch.matmul(h_proj, mem_proj.transpose(-2, -1))
        attn = attn / (self.head_dim ** 0.5)
        gate = torch.sigmoid(self.gate(h).transpose(1, 2).unsqueeze(-1))
        attn = attn * gate
        attn_weights = torch.softmax(attn, dim=-1)
        out = torch.matmul(attn_weights, mem_proj)
        out = out.transpose(1, 2).reshape(b, t, d)
        return out


class LTSIndexCompressor:
    def __init__(self, d_mem: int, compression_ratio: float = 0.5):
        self.d_mem = d_mem
        self.ratio = compression_ratio
        self.encoder = nn.Sequential(
            nn.Linear(d_mem, int(d_mem * compression_ratio)),
            nn.ReLU(),
        )
        self.decoder = nn.Linear(int(d_mem * compression_ratio), d_mem)

    def compress(self, lts: torch.Tensor) -> torch.Tensor:
        return self.encoder(lts)

    def decompress(self, compressed: torch.Tensor) -> torch.Tensor:
        return self.decoder(compressed)

    def memory_saved(self, n_entries: int) -> float:
        original = n_entries * self.d_mem * 4
        compressed = n_entries * int(self.d_mem * self.ratio) * 4
        return (1 - compressed / original) * 100
