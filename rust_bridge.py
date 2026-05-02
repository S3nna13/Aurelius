import os
import sys
import logging
logger = logging.getLogger("rust_bridge")


try:
    import aurelius_memory
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

_rust_page_table = None


def get_page_table(capacity: int = 4096, gpu_budget_mb: int = 65536):
    global _rust_page_table
    if not HAS_RUST:
        return _PyFallbackPageTable(capacity)
    if _rust_page_table is None:
        _rust_page_table = aurelius_memory.MemoryPageTable(capacity, gpu_budget_mb)
    return _rust_page_table


def save_checkpoint(path: str, tensors: dict):
    if not HAS_RUST:
        return _py_save_checkpoint(path, tensors)
    writer = aurelius_memory.MmapCheckpointWriter(path)
    writer.open()
    for name, tensor in tensors.items():
        writer.write_tensor(name, tensor.flatten().tolist())
    writer.finalize()


def estimate_memory(d_model: int, d_ff: int, n_heads: int,
                    seq_len: int, batch_size: int, precision_bytes: int = 2) -> str:
    if HAS_RUST:
        return aurelius_memory.estimate_layer_memory(
            d_model, d_ff, n_heads, seq_len, batch_size, precision_bytes
        )
    return _py_estimate_memory(d_model, d_ff, n_heads, seq_len, batch_size, precision_bytes)


class _PyFallbackPageTable:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.pages = {}
        self.clock = 0

    def register_page(self, id: int, priority: float, size_bytes: int, on_gpu: bool) -> str:
        if len(self.pages) >= self.capacity:
            return f"full:{self.capacity}"
        self.pages[id] = {'priority': priority, 'size': size_bytes, 'gpu': on_gpu, 'access': 0, 'last': 0}
        return "ok"

    def access(self, id: int) -> str:
        if id not in self.pages:
            return "absent"
        self.pages[id]['access'] += 1
        self.pages[id]['last'] = self.clock
        self.clock += 1
        return "gpu" if self.pages[id]['gpu'] else "cpu"

    def promote_to_gpu(self, id: int) -> str:
        if id not in self.pages:
            return "absent"
        self.pages[id]['gpu'] = True
        return "promoted"

    def stats(self) -> str:
        gpu = sum(1 for p in self.pages.values() if p['gpu'])
        cpu = len(self.pages) - gpu
        return f"pages={len(self.pages)} gpu={gpu} cpu={cpu} (py fallback)"


def _py_save_checkpoint(path: str, tensors: dict):
    import torch
    torch.save(tensors, path)


def _py_estimate_memory(d_model, d_ff, n_heads, seq_len, batch_size, precision_bytes):
    attn_params = 4 * d_model * d_model
    ffn_params = 3 * d_model * d_ff
    total_params = attn_params + ffn_params
    param_bytes = total_params * precision_bytes
    total_mb = param_bytes / (1024 * 1024)
    return f"layer: {total_params//1000000}M params, {total_mb:.1f}MB weights (py estimate)"


def build_rust_crate():
    import subprocess
    crate_dir = os.path.join(os.path.dirname(__file__), 'rust_memory')
    result = subprocess.run(
        ['cargo', 'build', '--release'],
        cwd=crate_dir,
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print("Rust crate built successfully")
        so_files = []
        for root, dirs, files in os.walk(os.path.join(crate_dir, 'target', 'release')):
            for f in files:
                if '.so' in f or '.dylib' in f:
                    so_files.append(os.path.join(root, f))
        return so_files
    else:
        print(f"Rust build failed: {result.stderr[:500]}")
        return []
