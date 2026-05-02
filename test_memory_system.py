import torch
import sys
sys.path.insert(0, '/Users/christienantonio/aurelius')
from async_memory import AsyncConsolidationPipeline, ConsolidationTask, PagedLTSMemory
from deduplication import CosineDeduplicator, L2Deduplicator, PriorityProportionalAllocator
from prefetch_router import PredictiveMemoryPrefetcher, SparseLTSRouter, PredictiveAttentionRouter, LTSIndexCompressor
from adaptive_precision import AdaptivePrecisionManager, FP8LTSMemory, TieredMemoryBank

B, D = 2, 64

def test_async_consolidation_shapes():
    pipeline = AsyncConsolidationPipeline(d_mem=D, lts_capacity=8, n_layers=4, device='cpu')
    slots = torch.randn(B, 16, D)
    surprise = torch.rand(B, 16)
    task = ConsolidationTask(layer=0, episodic_slots=slots, surprise_scores=surprise, timestamp=1.0, priority=0.5)
    result = pipeline._consolidate(task)
    assert result['consolidated'].shape == (B, 8, D)
    assert result['new_entries'] == 8

def test_paged_lts_write_read():
    mem = PagedLTSMemory(d_mem=D, capacity=8, n_pages=2, page_size=4, device='cpu')
    query = torch.randn(B, 10, D)
    keys = torch.randn(1, 4, D)
    values = torch.randn(1, 4, D)
    imp = torch.rand(1, 4, 1)
    mem.write(keys, values, imp)
    out = mem.read(query, top_k=4)
    assert out.shape == (B, 10, D)

def test_cosine_deduplicator():
    entries = torch.randn(1, 10, D)
    entries[0, 3] = entries[0, 1] + 0.01
    entries[0, 7] = entries[0, 1] + 0.01
    dedup = CosineDeduplicator(threshold=0.9, window=128)
    dups = dedup.find_duplicates(entries)
    merged = dedup.merge(entries, dups)
    assert merged.shape[1] <= entries.shape[1]
    assert isinstance(dups, list)

def test_l2_deduplicator():
    entries = torch.randn(1, 10, D)
    entries[0, 5] = entries[0, 2]
    d = L2Deduplicator(eps=0.01)
    deduped, mapping = d.deduplicate(entries)
    assert deduped.shape == (1, 9, D)
    assert mapping.shape == (10,)
    assert (mapping == -1).sum() == 1

def test_priority_allocator():
    alloc = PriorityProportionalAllocator(total_capacity=100, n_layers=4)
    alloc.update_priority(layer=2, surprise=0.9)
    alloc.update_priority(layer=2, surprise=0.9)
    alloc.update_priority(layer=2, surprise=0.9)
    slots = alloc.allocate()
    assert len(slots) == 4
    assert sum(slots) == 100
    assert slots[2] >= slots[0]

def test_predictive_prefetcher():
    pref = PredictiveMemoryPrefetcher(d_model=128, d_mem=D, n_layers=4)
    for _ in range(10):
        pref.observe(layer=1, surprise=torch.tensor(0.8), hidden=torch.randn(1, D))
    pred = pref.predict_next_surprise(layer=1)
    assert 0.0 <= pred <= 1.0
    lts = torch.randn(1, 100, D)
    q = torch.randn(B, 5, D)
    idx = pref.get_prefetch_indices(lts, q, top_k=16)
    assert idx.shape == (B, 5, 16)

def test_sparse_router_shapes():
    router = SparseLTSRouter(d_model=D, d_mem=D, top_k=16, n_experts=4)
    h = torch.randn(B, 8, D)
    lts = torch.randn(1, 128, D)
    result, weights = router(h, lts)
    assert result.shape == (B, 8, D)
    assert weights.shape == (B, 8, 4)

def test_predictive_attention_router():
    par = PredictiveAttentionRouter(d_model=128, n_heads=8, n_memory_slots=64)
    h = torch.randn(B, 8, 128)
    mem = torch.randn(1, 64, 128)
    out = par(h, mem)
    assert out.shape == (B, 8, 128)

def test_adaptive_precision():
    apm = AdaptivePrecisionManager()
    assert apm.get_dtype('working_memory') == torch.float16
    t = torch.randn(4, D, dtype=torch.float32)
    qt = apm.quantize_to_tier(t, 'graph_edges')
    assert qt.dtype == torch.int8
    assert qt.shape == (4, D)
    apm.auto_tune('working_memory', error_rate=0.1)

def test_fp8_memory():
    fp8 = FP8LTSMemory(d_mem=D, capacity=128)
    data = torch.randn(1, 5, D)
    indices = torch.tensor([10, 20, 30, 40, 50])
    fp8.store(data, indices)
    query = torch.randn(B, 8, D)
    out = fp8.retrieve(query, top_k=16)
    assert out.shape == (B, 8, D)

def test_tiered_memory_bank():
    bank = TieredMemoryBank(d_mem=D, capacities={'core': 32, 'working': 128, 'archive': 512})
    assert bank.route_to_tier(torch.tensor(0.85)) == 'core'
    assert bank.route_to_tier(torch.tensor(0.15)) == 'archive'
    data = torch.randn(1, D)
    bank.write('test', data, torch.tensor(0.9))
    mb = bank.total_memory_mb()
    assert 'core' in mb and 'working' in mb and 'archive' in mb
