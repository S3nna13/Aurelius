#!/usr/bin/env python3
"""Benchmark the hierarchical KV‑cache eviction path.

Runs a synthetic cache with a realistic shape, calls ``evict`` repeatedly
and reports average latency.
"""
import time
import torch

from src.model.transformer import HierarchicalKVCache  # type: ignore

def bench(batch=8, n_head=12, seq_len=2048, head_dim=64, iterations=1000):
    cache = HierarchicalKVCache(n_head=n_head, n_kv_heads=n_head, head_dim=head_dim, max_seq_len=seq_len)
    # Warm‑up
    for _ in range(10):
        cache.evict(torch.tensor([seq_len // 2]))
    start = time.perf_counter()
    for _ in range(iterations):
        cache.evict(torch.tensor([seq_len // 2]))
    elapsed = time.perf_counter() - start
    print(f"{iterations} evicts in {elapsed:.3f}s => {iterations/elapsed:.2f} evicts/sec")

if __name__ == "__main__":
    bench()
