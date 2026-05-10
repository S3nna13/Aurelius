import torch
import torch.nn as nn
import torch.nn.functional as F


class KVCacheQuantizer:
    def __init__(self, bits: int = 8, block_size: int = 64):
        self.bits = bits
        self.block_size = block_size
        self.qmax = 2 ** (bits - 1) - 1
        self.qmin = -(2 ** (bits - 1))

    def quantize(self, tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        shape = tensor.shape
        flattened = tensor.view(-1, self.block_size)
        amax = flattened.abs().amax(dim=-1, keepdim=True).clamp(min=1e-6)
        scale = amax / self.qmax
        quantized = (flattened / scale).round().clamp(self.qmin, self.qmax).to(torch.int8)
        return quantized.view(shape), scale.view(shape[0], -1)

    def dequantize(self, quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        shape = quantized.shape
        flattened = quantized.view(-1, self.block_size).float()
        s = scale.view(-1, 1)
        return (flattened * s).view(shape)

    def quantize_kv_cache(self, key: torch.Tensor, value: torch.Tensor
                          ) -> tuple[tuple, tuple]:
        k_q, k_s = self.quantize(key)
        v_q, v_s = self.quantize(value)
        return (k_q, k_s), (v_q, v_s)

    def dequantize_kv_cache(self, k_quant: tuple, v_quant: tuple) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.dequantize(*k_quant)
        v = self.dequantize(*v_quant)
        return k, v


class PagedAttentionCache:
    def __init__(self, n_layers: int, n_heads: int, head_dim: int,
                 max_blocks: int = 2048, block_size: int = 16):
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.max_blocks = max_blocks
        self.block_size = block_size
        self.free_blocks = list(range(max_blocks))
        self.block_tables = [[] for _ in range(n_layers)]
        self.kv_data = torch.zeros(max_blocks, block_size, 2, n_heads, head_dim,
                                   dtype=torch.float16, device='cuda' if torch.cuda.is_available() else 'cpu')

    def alloc_blocks(self, n: int) -> list[int]:
        if n > len(self.free_blocks):
            raise MemoryError(f"requested {n} blocks, only {len(self.free_blocks)} free")
        allocated = self.free_blocks[:n]
        self.free_blocks = self.free_blocks[n:]
        return allocated

    def release_blocks(self, blocks: list[int]):
        self.free_blocks.extend(blocks)
        self.free_blocks.sort()

    def write_block(self, layer: int, block_idx: int, key: torch.Tensor, value: torch.Tensor):
        self.kv_data[block_idx, :key.shape[0], 0] = key
        self.kv_data[block_idx, :value.shape[0], 1] = value

    def read_block(self, layer: int, block_idx: int, seq_pos: int, seq_len: int
                   ) -> tuple[torch.Tensor, torch.Tensor]:
        k = self.kv_data[block_idx, :seq_len, 0]
        v = self.kv_data[block_idx, :seq_len, 1]
        return k, v


class CompressedContextBuffer:
    def __init__(self, d_model: int, max_compressed_tokens: int = 512):
        self.d_model = d_model
        self.max_tokens = max_compressed_tokens
        self.buffer = []
        self.compressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, d_model),
        )

    def add(self, hidden_states: torch.Tensor):
        self.buffer.append(hidden_states.detach().cpu())
        if len(self.buffer) > self.max_tokens:
            self._compress()

    def _compress(self):
        stacked = torch.stack(self.buffer)
        compressed = self.compressor(stacked)
        keep = compressed[::2]
        self.buffer = [t for t in keep]

    def get_context(self, device: str = 'cuda') -> torch.Tensor:
        if not self.buffer:
            return torch.empty(0, self.d_model, device=device)
        return torch.stack(self.buffer).to(device)


class MemoryBudgetTracker:
    def __init__(self, target_mb: int = 24000):
        self.target_bytes = target_mb * 1024 * 1024
        self.peak_bytes = 0
        self.allocations = {}

    @torch.no_grad()
    def snapshot(self, label: str = ""):
        if not torch.cuda.is_available():
            return 0
        torch.cuda.synchronize()
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        self.peak_bytes = max(self.peak_bytes, allocated)
        self.allocations[label] = {'allocated_mb': allocated / 1024 / 1024,
                                    'reserved_mb': reserved / 1024 / 1024,
                                    'peak_mb': self.peak_bytes / 1024 / 1024}
        return allocated

    def report(self) -> str:
        lines = []
        for label, stats in self.allocations.items():
            lines.append(f"  {label}: {stats['allocated_mb']:.0f}MB (reserved {stats['reserved_mb']:.0f}MB)")
        lines.append(f"  Peak: {self.peak_bytes / 1024 / 1024:.0f}MB")
        return "\n".join(lines)

    def can_fit(self, additional_mb: int) -> bool:
        current = self.allocations.get('latest', {}).get('allocated_mb', 0)
        return (current + additional_mb) * 1024 * 1024 < self.target_bytes
