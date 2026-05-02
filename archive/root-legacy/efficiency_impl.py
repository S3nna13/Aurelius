import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class TiledFlashAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, block_size: int = 128):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.block_size = block_size
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, block_size: int | None = None, return_kv: bool = False) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bs = block_size or self.block_size
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if t <= bs:
            out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            out = self.out(out.transpose(1, 2).reshape(b, t, d))
            if return_kv:
                return out, k, v
            return out

        out_blocks = []
        for i in range(0, t, bs):
            q_block = q[:, :, i:i + bs]
            m_i = torch.full((b, self.n_heads, bs), float('-inf'), device=x.device)
            l_i = torch.zeros((b, self.n_heads, bs), device=x.device)
            acc = torch.zeros((b, self.n_heads, bs, self.head_dim), device=x.device)
            for j in range(0, t, bs):
                k_block = k[:, :, j:j + bs]
                v_block = v[:, :, j:j + bs]
                s = torch.matmul(q_block, k_block.transpose(-2, -1)) / math.sqrt(self.head_dim)
                if j > i:
                    s = s.masked_fill(torch.ones_like(s, dtype=torch.bool), float('-inf'))
                elif j == i:
                    s = s.masked_fill(torch.ones_like(s, dtype=torch.bool).triu(diagonal=1), float('-inf'))
                m_ij = torch.maximum(m_i, s.amax(dim=-1, keepdim=True).squeeze(-1))
                p = torch.exp(s - m_ij.unsqueeze(-1))
                l_ij = p.sum(dim=-1)
                acc = acc * torch.exp(m_i - m_ij).unsqueeze(-1) + torch.matmul(p, v_block)
                m_i, l_i = m_ij, l_ij
            out_blocks.append(acc / l_i.unsqueeze(-1))
        out = torch.cat(out_blocks, dim=2)
        out = self.out(out.transpose(1, 2).reshape(b, t, d))
        if return_kv:
            return out, k, v
        return out


class PagedKVManager(nn.Module):
    def __init__(self, n_layers: int, n_heads: int, head_dim: int,
                 block_size: int = 16, max_blocks: int = 4096, device: str = 'cuda'):
        super().__init__()
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.block_size = block_size
        self.device = device
        self.free_blocks = list(range(max_blocks))
        self.block_tables = {l: {} for l in range(n_layers)}
        self.register_buffer('kv_data', torch.zeros(max_blocks, block_size, 2, n_heads, head_dim))

    def alloc(self, n: int) -> list[int]:
        taken = self.free_blocks[:n]
        self.free_blocks = self.free_blocks[n:]
        return taken

    def free(self, block_ids: list[int]):
        self.free_blocks.extend(block_ids)
        self.free_blocks.sort()

    def write(self, layer: int, block_id: int, pos: int, key: torch.Tensor, value: torch.Tensor):
        self.kv_data[block_id, pos % self.block_size, 0] = key
        self.kv_data[block_id, pos % self.block_size, 1] = value

    def read(self, layer: int, block_id: int, seq_len: int) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.kv_data[block_id, :seq_len, 0]
        v = self.kv_data[block_id, :seq_len, 1]
        return k, v

    def get_kv_for_layer(self, layer: int, seq_id: int) -> tuple[torch.Tensor, torch.Tensor]:
        blocks = self.block_tables[layer].get(seq_id, [])
        ks, vs = [], []
        for bid in blocks:
            k, v = self.read(layer, bid, self.block_size)
            ks.append(k)
            vs.append(v)
        return torch.cat(ks, dim=0), torch.cat(vs, dim=0)


class StreamingCache(nn.Module):
    def __init__(self, d_model: int, window_size: int = 2048, n_sink: int = 4):
        super().__init__()
        self.window_size = window_size
        self.n_sink = n_sink
        self.proj = nn.Linear(d_model, d_model)
        self.gate = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor, cache: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        b, t, d = x.shape
        if cache is None:
            return x, x.detach()
        if t <= self.n_sink:
            return x, torch.cat([cache, x.detach()], dim=1)
        sink = cache[:, :self.n_sink].detach()
        window = torch.cat([cache[:, -self.window_size + t:], x], dim=1).detach() if cache.shape[1] > self.window_size else torch.cat([cache, x], dim=1)
        new_cache = torch.cat([sink, window[:, -self.window_size:] if window.shape[1] > self.window_size else window], dim=1)
        fused = self.gate * self.proj(sink.mean(dim=1, keepdim=True)) + x
        return fused, new_cache.detach()

    def generate_streaming(self, model, prompt: torch.Tensor, max_new: int = 100) -> torch.Tensor:
        cache = None
        out = prompt
        for _ in range(max_new):
            with torch.no_grad():
                h = model.token_embedding(out)
                h, cache = self(h, cache)
                logits = model.lm_head(h[:, -1:])
            next_tok = logits.argmax(dim=-1)
            out = torch.cat([out, next_tok], dim=-1)
        return out


class ZeROOptimizer:
    def __init__(self, model: nn.Module, optimizer_class=torch.optim.AdamW,
                 lr: float = 3e-4, weight_decay: float = 0.1, cpu_offload: bool = False):
        self.cpu_offload = cpu_offload
        self.optimizer = optimizer_class(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.param_list = list(model.parameters())

    def step(self, loss: torch.Tensor) -> torch.Tensor:
        loss.backward()
        if self.cpu_offload:
            for p in self.param_list:
                if p.grad is not None:
                    p.grad = p.grad.cpu()
                    p.data = p.data.cpu()
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.cpu()
        self.optimizer.step()
        self.optimizer.zero_grad()
        if self.cpu_offload:
            device = self.param_list[0].data.device if len(self.param_list) > 0 else torch.device('cuda')
            for p in self.param_list:
                p.data = p.data.to(device)
            for state in self.optimizer.state.values():
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        return loss


class DistributedTrainingManager(nn.Module):
    def __init__(self, d_model: int, d_ff: int, n_heads: int, n_pp_stages: int = 2):
        super().__init__()
        self.n_pp_stages = n_pp_stages
        self.column_parallel = nn.Linear(d_model, d_ff, bias=False)
        self.row_parallel = nn.Linear(d_ff, d_model, bias=False)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)

    def tensor_parallel_forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.column_parallel(x)
        h = F.relu(h)
        return self.row_parallel(h)

    def pipeline_parallel_forward(self, micro_batches: list[torch.Tensor]) -> list[torch.Tensor]:
        outputs = []
        for stage in range(self.n_pp_stages):
            stage_out = []
            for mb in micro_batches:
                if stage == 0:
                    h = self.attn(mb, mb, mb)[0]
                else:
                    h = self.tensor_parallel_forward(mb)
                stage_out.append(h)
            micro_batches = stage_out
        return micro_batches
