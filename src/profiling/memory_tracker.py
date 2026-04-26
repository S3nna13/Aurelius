from __future__ import annotations

import gc

import torch
import torch.nn as nn


def format_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if abs(n) < 1024:
            return f"{n:.2f}{unit}"
        n //= 1024
    return f"{n:.2f}TB"


class MemoryTracker:
    def snapshot_params(self, model: nn.Module) -> dict[str, int]:
        seen_ids: set[int] = set()
        total_numel = 0
        total_bytes = 0
        for p in model.parameters():
            pid = id(p)
            if pid not in seen_ids:
                seen_ids.add(pid)
                total_numel += p.numel()
                total_bytes += p.numel() * p.element_size()
        for b in model.buffers():
            bid = id(b)
            if bid not in seen_ids:
                seen_ids.add(bid)
                total_numel += b.numel()
                total_bytes += b.numel() * b.element_size()
        return {"numel": total_numel, "bytes": total_bytes}

    def module_breakdown(self, model: nn.Module) -> dict[str, dict[str, int]]:
        breakdown: dict[str, dict[str, int]] = {}
        for name, module in model.named_modules():
            if list(module.parameters(recurse=False)):
                numel = sum(p.numel() for p in module.parameters(recurse=False))
                bytes_ = sum(p.numel() * p.element_size() for p in module.parameters(recurse=False))
                breakdown[name or module.__class__.__name__] = {"numel": numel, "bytes": bytes_}
        return breakdown

    def peak_diff(self) -> tuple[int, int] | None:
        gc.collect()
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated(), torch.cuda.max_memory_allocated()
        return None


MEMORY_TRACKER = MemoryTracker()
