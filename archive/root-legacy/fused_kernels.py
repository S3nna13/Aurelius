import torch
import torch.nn as nn

try:
    from flash_attn import flash_attn_func
    HAS_FLASH_3 = True
except ImportError:
    HAS_FLASH_3 = False


class FlashAttention3Wrapper(nn.Module):
    def __init__(self, d_model: int, n_heads: int, causal: bool = True):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        b, t, d = x.shape
        qkv = self.qkv(x).reshape(b, t, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        if HAS_FLASH_3 and x.is_cuda and self.head_dim <= 256:
            from flash_attn import flash_attn_func
            q = q.transpose(1, 2).contiguous()
            k = k.transpose(1, 2).contiguous()
            v = v.transpose(1, 2).contiguous()
            out = flash_attn_func(q, k, v, dropout_p=0.0, softmax_scale=self.head_dim ** -0.5,
                                  causal=self.causal)
            out = out.transpose(1, 2).reshape(b, t, d)
        else:
            from nn_utils import apply_rotary
            q = apply_rotary(q, cos, sin)
            k = apply_rotary(k, cos, sin)
            attn = (q @ k.transpose(-2, -1)) / (self.head_dim ** 0.5)
            if self.causal:
                mask = torch.tril(torch.ones(t, t, device=x.device)).view(1, 1, t, t)
                attn = attn.masked_fill(mask == 0, float('-inf'))
            attn = torch.softmax(attn, dim=-1)
            out = (attn @ v).transpose(1, 2).reshape(b, t, d)

        return self.out(out)

    def flops_per_step(self, seq_len: int) -> int:
        return 4 * seq_len * self.d_model * self.d_model


class FusedMemoryKernel:
    @staticmethod
    def fused_read_write(query: torch.Tensor, memory: torch.Tensor,
                         importance: torch.Tensor) -> torch.Tensor:
        scores = query @ memory.transpose(-2, -1)
        scores = scores / (memory.shape[-1] ** 0.5)
        attn = torch.softmax(scores, dim=-1)
        read = attn @ memory
        if importance is not None:
            imp = torch.sigmoid(importance)
            write = imp.unsqueeze(-1) * query
            memory = memory + 0.01 * (attn.transpose(-2, -1) @ write)
        return read, memory

    @staticmethod
    def fuse_surprise_consolidation(hidden: torch.Tensor, episodic: torch.Tensor,
                                     lts: torch.Tensor) -> tuple:
        surprise = torch.sigmoid(hidden @ hidden.transpose(-2, -1)).mean(dim=-1, keepdim=True)
        normed = episodic / (episodic.norm(dim=-1, keepdim=True) + 1e-8)
        sim = normed @ normed.transpose(-2, -1)
        adj = (sim > 0.65).float()
        clusters = (adj @ episodic) / adj.sum(dim=-1, keepdim=True).clamp(min=1)
        weighted = clusters * surprise
        top_k = min(lts.shape[1], weighted.shape[1])
        vals, idx = surprise.squeeze(-1).topk(top_k, dim=-1)
        lts.data[0, :top_k] = weighted[0, idx[0]]
        return lts, surprise


class CUDAGraphMemoryWrapper:
    def __init__(self, module: nn.Module, n_warmup: int = 5):
        self.module = module
        self.graph = None
        self.static_inputs = None
        self.static_output = None
        self.warm = False

    def warmup(self, sample_input: torch.Tensor):
        if not torch.cuda.is_available():
            return
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(self.n_warmup):
                _ = self.module(sample_input)
        torch.cuda.current_stream().wait_stream(s)

        self.graph = torch.cuda.CUDAGraph()
        self.static_inputs = sample_input
        self.static_output = torch.zeros_like(sample_input)
        with torch.cuda.graph(self.graph):
            self.static_output = self.module(self.static_inputs)
        self.warm = True

    def replay(self, x: torch.Tensor) -> torch.Tensor:
        if not self.warm:
            return self.module(x)
        self.static_inputs.copy_(x)
        self.graph.replay()
        return self.static_output.clone()


class KernelRegistry:
    def __init__(self):
        self.kernels = {}

    def register(self, name: str, fn):
        self.kernels[name] = fn

    def dispatch(self, name: str, *args, **kwargs):
        kernel = self.kernels.get(name)
        if kernel is None:
            raise ValueError(f"Kernel {name} not found")
        return kernel(*args, **kwargs)

    def autotune(self, name: str, *args, time_fn=None):
        kernel = self.kernels.get(name)
        if kernel is None:
            return None
        if time_fn is None:
            import time
            time_fn = time.time
        start = time_fn()
        result = kernel(*args)
        elapsed = time_fn() - start
        return result, elapsed
