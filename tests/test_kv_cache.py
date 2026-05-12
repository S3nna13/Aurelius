from __future__ import annotations

import pytest

pytest.skip("module removed during reorganization", allow_module_level=True)


import torch
from hierarchical_kv_cache import HierarchicalKVCache


def test_hierarchical_kv_cache_writes_and_evicts():
    torch.manual_seed(0)
    cache = HierarchicalKVCache(
        d_model=8,
        n_heads=2,
        head_dim=4,
        cap1=2,
        cap2=4,
        cap3=4,
        max_batch=1,
    )
    key = torch.randn(1, 2, 2, 4)
    value = torch.randn(1, 2, 2, 4)
    hidden = torch.randn(1, 1, 8)

    cache.write(key, value, hidden)
    assert cache.t1_n.item() == 2

    cache.write(key[:, :, :1], value[:, :, :1], hidden)
    assert cache.t1_n.item() == 2
    assert cache.t2_n.item() >= 1
