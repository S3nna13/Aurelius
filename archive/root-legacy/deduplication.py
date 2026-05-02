import torch
import logging
logger = logging.getLogger("deduplication")



class CosineDeduplicator:
    def __init__(self, threshold: float = 0.92, window: int = 128):
        self.threshold = threshold
        self.window = window

    def find_duplicates(self, entries: torch.Tensor) -> list[tuple[int, int]]:
        n = entries.shape[1]
        dups = []
        normed = entries / (entries.norm(dim=-1, keepdim=True) + 1e-8)
        for i in range(n):
            start = max(0, i - self.window)
            for j in range(start, i):
                sim = (normed[0, i] * normed[0, j]).sum()
                if sim > self.threshold:
                    dups.append((j, i))
                    break
        return dups

    def merge(self, entries: torch.Tensor, duplicates: list[tuple[int, int]]) -> torch.Tensor:
        keep = set(range(entries.shape[1]))
        merged_map = {}
        for src, dst in duplicates:
            if dst in keep:
                keep.remove(dst)
                merged_map.setdefault(src, []).append(dst)
        keep_list = list(keep)
        keep_index_map = {v: i for i, v in enumerate(keep_list)}
        out = entries[:, keep_list]
        for src, dsts in merged_map.items():
            idx = keep_index_map[src]
            total = out[0, idx] + sum(entries[0, d] for d in dsts)
            out[0, idx] = total / (len(dsts) + 1)
        return out

    def compressed_size(self, original: int, duplicates: list) -> int:
        removed = len(set(d for _, d in duplicates))
        return original - removed


class L2Deduplicator:
    def __init__(self, eps: float = 0.01):
        self.eps = eps

    def deduplicate(self, entries: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n = entries.shape[1]
        normed = entries / (entries.norm(dim=-1, keepdim=True) + 1e-8)
        dist = torch.cdist(normed, normed, p=2)
        mask = dist < self.eps
        triu = torch.triu(mask, diagonal=1)
        dup_idx = triu.nonzero(as_tuple=False)
        keep = torch.ones(n, dtype=torch.bool)
        for _, _, j in dup_idx:
            keep[j] = False
        deduped = entries[:, keep]
        mapping = keep.cumsum(0) - 1
        mapping[~keep] = -1
        return deduped, mapping


class TemporalDeduplicator:
    def __init__(self, window: int = 64, decay: float = 0.9):
        self.window = window
        self.decay = decay
        self.fingerprints = []

    def fingerprint(self, entry: torch.Tensor) -> torch.Tensor:
        signs = entry.sign().flatten()
        if signs.numel() < 128:
            signs = torch.nn.functional.pad(signs, (0, 128 - signs.numel()))
        return signs[:128]

    def is_redundant(self, entry: torch.Tensor) -> bool:
        fp = self.fingerprint(entry)
        for past_fp, _ in self.fingerprints[-self.window:]:
            sim = (fp * past_fp).sum() / max(fp.norm() * past_fp.norm(), 1e-8)
            if sim > 0.95:
                return True
        return False

    def record(self, entry: torch.Tensor, priority: float):
        fp = self.fingerprint(entry)
        self.fingerprints.append((fp, priority))
        if len(self.fingerprints) > self.window * 4:
            scored = sorted(
                self.fingerprints, key=lambda x: x[1], reverse=True
            )[:self.window]
            self.fingerprints = scored


class PriorityProportionalAllocator:
    def __init__(self, total_capacity: int, n_layers: int, alpha: float = 0.3):
        self.total = total_capacity
        self.n_layers = n_layers
        self.alpha = alpha
        self.layer_priority = torch.ones(n_layers) / n_layers

    def update_priority(self, layer: int, surprise: float):
        smoothed = (1 - self.alpha) * self.layer_priority[layer] + self.alpha * surprise
        self.layer_priority[layer] = smoothed.item()
        self.layer_priority = self.layer_priority / self.layer_priority.sum()

    def allocate(self) -> list[int]:
        base = self.total // self.n_layers
        extra_pool = self.total - base * self.n_layers
        allocations = [base] * self.n_layers
        prio_sorted = self.layer_priority.argsort(descending=True)
        for i in range(extra_pool):
            allocations[prio_sorted[i % self.n_layers].item()] += 1
        return allocations

    def report(self) -> str:
        alloc = self.allocate()
        lines = ["Priority-Proportional Allocation:"]
        for i in range(self.n_layers):
            lines.append(f"  Layer {i}: priority={self.layer_priority[i]:.3f}, "
                        f"allocated={alloc[i]} slots")
        return "\n".join(lines)


class MemoryAwareGradientAccumulator:
    def __init__(self, target_memory_mb: int = 48000, accumulation_steps: int = 8):
        self.target_bytes = target_memory_mb * 1024 * 1024
        self.accumulation_steps = accumulation_steps
        self.current_step = 0
        self.peak_memory = 0

    def should_accumulate(self, current_memory_mb: float) -> bool:
        over_budget = current_memory_mb * 1024 * 1024 > self.target_bytes
        return over_budget or (self.current_step % self.accumulation_steps != 0)

    def step(self, memory_mb: float) -> bool:
        self.current_step += 1
        self.peak_memory = max(self.peak_memory, memory_mb)
        return self.should_accumulate(memory_mb)


class LZ4MemoryCompressor:
    def __init__(self, compression_level: int = 6):
        self.level = compression_level
        self.compressed_pages = {}

    def compress_page(self, page_id: int, tensor: torch.Tensor) -> int:
        if not hasattr(self, '_lz4'):
            import lz4.frame
            self._lz4 = lz4.frame
        data = tensor.cpu().numpy().tobytes()
        compressed = self._lz4.compress(data)
        self.compressed_pages[page_id] = (compressed, tensor.dtype)
        return len(compressed)

    def decompress_page(self, page_id: int, shape: tuple) -> torch.Tensor:
        import lz4.frame
        import numpy as np
        entry = self.compressed_pages.get(page_id)
        if entry is None:
            return None
        compressed, dtype = entry
        data = lz4.frame.decompress(compressed)
        array = np.frombuffer(data, dtype=str(dtype)).reshape(shape)
        return torch.from_numpy(array)

    def compression_ratio(self, page_id: int, original_size: int) -> float:
        if page_id not in self.compressed_pages:
            return 1.0
        return len(self.compressed_pages[page_id]) / original_size
