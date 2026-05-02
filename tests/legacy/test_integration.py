import torch
import sys
sys.path.insert(0, '/Users/christienantonio/aurelius')


def test_brain_layer_v1():
    from brain_layer import NeuralBrainLayer
    b = NeuralBrainLayer(d_brain=256, d_model=768)
    x = torch.randn(1, 32, 768)
    out = b(x)
    assert out['output'].shape == (1, 256)
    assert out['steps'] >= 1
    assert 'epistemic' in out['uncertainty']
    assert 'summary' in out['reflection']
    print("  v1 NeuralBrainLayer: OK")


def test_brain_integrated():
    from brain_integrated import IntegratedNeuralBrain
    b = IntegratedNeuralBrain(d_brain=768, d_model=768)
    x = torch.randn(1, 32, 768)
    out = b(x)
    assert out['output'].shape == (1, 32, 768), f"integrated output: {out['output'].shape}"
    assert 0.01 <= out['confidence'] <= 0.99, f"confidence clamped: {out['confidence']}"
    assert out['steps'] >= 1
    stages = [t.get('stage', '?') for t in out['trajectory']]
    assert 'quiet_star' in stages
    assert 'tot' in stages
    assert 'star' in stages
    assert 'verify' in stages
    assert 'reflect' in stages
    assert 'epistemic' in out['uncertainty']
    print("  v2 IntegratedNeuralBrain: OK")


def test_reasoning_components():
    from reasoning_paper_impl import TreeOfThoughtSearcher
    problem = torch.randint(0, 1000, (1, 8))
    t = TreeOfThoughtSearcher(d_model=768, vocab_size=1000)
    r = t(problem)
    assert 'best_path' in r
    assert 'value_scores' in r
    print("  Reasoning components (ToT): OK")


def test_memory_moe_components():
    from memory_moe_impl import (
        NoisyTopKRouter, ExpertChoiceRouter,
        CompressiveMemory, KNNAttentionLayer, MoEMemoryLayer
    )
    x = torch.randn(2, 16, 768)
    nr = NoisyTopKRouter(768, 4)
    wd, ei, la = nr(x)
    assert la >= 0
    ec = ExpertChoiceRouter(768, 4)
    ec_out = ec(x)  # returns 6-tuple
    assert len(ec_out) >= 1
    cm = CompressiveMemory(768, 64, 256)
    state = (torch.randn(2, 64, 768), torch.randn(2, 256, 768))
    cm_out, cm_loss, cm_state = cm(x, state)
    assert cm_out.shape == x.shape
    kn = KNNAttentionLayer(768, 8)
    assert kn(x).shape == x.shape
    mm = MoEMemoryLayer(768, 8, 64)
    mm_out, mm_aux, mm_state = mm(x)
    assert mm_out.shape == x.shape
    assert isinstance(mm_aux, torch.Tensor)
    print("  Memory/MoE components (5/5): OK")


def test_alignment_components():
    from alignment_impl import DPOLoss, ConstitutionalClassifier, ProcessRewardModel
    dpo = DPOLoss()
    chosen = torch.tensor([[0.1, 0.2, 0.3]])
    rejected = torch.tensor([[0.3, 0.2, 0.1]])
    l, a = dpo(chosen, rejected, chosen, rejected)
    assert isinstance(l, torch.Tensor)
    assert isinstance(a, torch.Tensor)
    cc = ConstitutionalClassifier(768)
    x2d = torch.randn(2, 768)
    vs, vt = cc(x2d)
    assert vs.shape == (2, 16)
    prm = ProcessRewardModel(768)
    steps = torch.randn(2, 8, 768)
    r = prm(steps)
    assert r.shape == (2, 8)
    print("  Alignment components (DPO, Constitutional, PRM): OK")


def test_efficiency_components():
    from efficiency_impl import TiledFlashAttention, PagedKVManager, StreamingCache
    x = torch.randn(2, 16, 768)
    ta = TiledFlashAttention(768, 8)
    assert ta(x).shape == x.shape
    pkm = PagedKVManager(4, 8, 96, device='cpu')
    blocks = pkm.alloc(4)
    assert len(blocks) == 4
    sc = StreamingCache(768)
    o, c = sc(x, x.detach())
    assert o.shape == x.shape
    assert c.shape[1] <= 2052
    print("  Efficiency components (Flash, Paged, Stream): OK")


def test_memory_core():
    from memory_core import AurelianMemoryCore
    mc = AurelianMemoryCore(d_model=768, d_mem=256, episodic_slots=64, lts_capacity=128)
    x = torch.randn(2, 16, 768)
    out = mc(x)
    assert out.shape == x.shape
    print("  AurelianMemoryCore: OK")


def test_agent_components():
    from agent_core import ToolFormerAdapter, PlanningModule, CriticHead, ValueHead
    from skills import SkillRegistry
    from agent_loop import AgentLoopController, AgentMemoryBridge, ExperienceReplayBuffer
    ta = ToolFormerAdapter(768, 8, 16)
    x = torch.randn(2, 16, 768)
    h, call = ta(x)
    assert h.shape == x.shape
    pm = PlanningModule(768, n_simulations=4, max_depth=4)
    plan, val, tree = pm(x)
    assert plan.shape == (2, 768)
    ch = CriticHead(768)
    score, suggestion = ch(x, x)
    assert score.shape == (2,)
    sr = SkillRegistry(768, skill_dim=64, max_skills=128)
    hs, skill = sr(x)
    alc = AgentLoopController(768, 8, 256)
    out = alc(x, full_cycle=True)
    assert 'hidden' in out
    assert 'value' in out
    amb = AgentMemoryBridge(768, 256, 64)
    mem = amb.read_from_memory(x, torch.randn(2, 64, 256))
    assert mem.shape == x.shape
    erb = ExperienceReplayBuffer(100)
    erb.push(x, x, 1.0, x, False)
    assert len(erb) == 1
    print("  Agent components (7/7): OK")


def test_kv_cache():
    from kv_cache_quant import KVCacheQuantizer, PagedAttentionCache
    k = torch.randn(2, 8, 32, 64)
    v = torch.randn(2, 8, 32, 64)
    q = KVCacheQuantizer(bits=8)
    (kq, ks), (vq, vs) = q.quantize_kv_cache(k, v)
    kd = q.dequantize(kq, ks)
    assert kd.shape == k.shape
    print("  KV Cache quant: OK")


def test_paged_lts():
    from async_memory import PagedLTSMemory
    p = PagedLTSMemory(256, 128, n_pages=16, page_size=8, device='cpu')
    query = torch.randn(1, 4, 256)
    out = p.read(query, top_k=16)
    assert out.shape == (1, 4, 256)
    print("  PagedLTSMemory: OK")


def test_fp8():
    from fp8_allreduce import FP8Compressor
    fc = FP8Compressor()
    x = torch.randn(2, 32, 768)
    q, s = fc.compress(x, dtype=torch.float8_e4m3fn)
    xd = fc.decompress(q, s)
    assert xd.shape == x.shape
    print("  FP8 compress/decompress: OK")


def test_speculative():
    from speculative_decoding import MemoryContextProjector
    mcp = MemoryContextProjector(768, 128)
    mem = torch.randn(2, 64, 768)
    proj = mcp(mem)
    assert proj.shape[-1] == 128
    print("  Speculative decoding projector: OK")


def test_moe():
    try:
        from moe_memory import MoELTSMemory
    except ImportError:
        from archive.moe_memory import MoELTSMemory
    moe = MoELTSMemory(256, 512, n_experts=4, top_k=2)
    x = torch.randn(2, 8, 256)
    out = moe(x, x)
    if isinstance(out, tuple):
        out = out[0]
    assert out.shape == x.shape
    print("  MoE Memory: OK")


def test_ntm():
    try:
        from ntm_memory import NTMMemory, NTMController
    except ImportError:
        from archive.ntm_memory import NTMMemory, NTMController
    nm = NTMMemory(64, 64)
    nc = NTMController(256, 64, 64)
    x = torch.randn(2, 256)
    mem_state = nm.reset(batch_size=2)
    read_vals, mem_out = nc(x, mem_state)
    assert read_vals.shape[-1] == 64
    print("  NTM Memory: OK")


def test_hierarchical_kv():
    from hierarchical_kv_cache import HierarchicalKVCache
    hkv = HierarchicalKVCache(256, 8, 32, cap1=8, cap2=16, cap3=32)
    k = torch.randn(1, 8, 4, 32)
    v = torch.randn(1, 8, 4, 32)
    hs = torch.randn(1, 1, 256)
    hkv.write(k[:,:,:4], v[:,:,:4], hs)
    k_read, v_read, tiers = hkv.read(n_tokens=8)
    assert k_read.shape[-1] == 32
    print("  Hierarchical KV Cache: OK")


def test_adaptive_precision():
    from adaptive_precision import AdaptivePrecisionManager
    apm = AdaptivePrecisionManager()
    x = torch.randn(2, 32)
    q = apm.quantize_to_tier(x, 'long_term_store')
    assert q.dtype == torch.float32
    print("  Adaptive Precision: OK")


def test_unified_manager():
    from unified_manager import UnifiedMemoryManager
    um = UnifiedMemoryManager(768, 256, 4, 128, 64, device='cpu', gpu_budget_mb=1024)
    x = torch.randn(2, 8, 768)
    s = torch.randn(2, 8)
    result = um.on_forward(0, x, x, x, s)
    assert 'should_accumulate' in result
    print("  UnifiedMemoryManager: OK")


def test_rust_bridge():
    from rust_bridge import get_page_table, _PyFallbackPageTable
    pt = get_page_table(64, 1024)
    r = pt.register_page(0, 0.8, 4096, True)
    assert r == "ok" or "full" not in r
    loc = pt.access(0)
    assert loc in ("gpu", "cpu", "absent")
    print("  Rust bridge (with Python fallback): OK")


def test_brain_layer_scale():
    from brain_layer import NeuralBrainLayer
    for d_brain, expected_range in [(512, (5, 30)), (1024, (20, 150)), (2048, (100, 600))]:
        b = NeuralBrainLayer(d_brain=d_brain, d_model=d_brain * 3)
        x = torch.randn(1, 16, d_brain * 3)
        out = b(x)
        total_m = sum(p.numel() for p in b.parameters()) / 1e6
        assert expected_range[0] <= total_m <= expected_range[1], \
            f"d_brain={d_brain}: {total_m:.1f}M outside [{expected_range[0]}, {expected_range[1]}]"
        assert out['output'].shape[-1] == d_brain
    print("  Brain layer scaling (512/1024/2048): OK")


if __name__ == '__main__':
    tests = [
        ("Brain Layer v1", test_brain_layer_v1),
        ("Brain Layer Integrated v2", test_brain_integrated),
        ("Reasoning Paper Components", test_reasoning_components),
        ("Memory/MoE Paper Components", test_memory_moe_components),
        ("Alignment Paper Components", test_alignment_components),
        ("Efficiency Paper Components", test_efficiency_components),
        ("Memory Core", test_memory_core),
        ("Agent Components", test_agent_components),
        ("KV Cache Quant", test_kv_cache),
        ("Paged LTS Memory", test_paged_lts),
        ("FP8 Compress", test_fp8),
        ("Speculative Decoding", test_speculative),
        ("MoE Memory", test_moe),
        ("NTM Memory", test_ntm),
        ("Hierarchical KV Cache", test_hierarchical_kv),
        ("Adaptive Precision", test_adaptive_precision),
        ("Unified Memory Manager", test_unified_manager),
        ("Rust Bridge", test_rust_bridge),
        ("Brain Layer Scaling", test_brain_layer_scale),
    ]

    passed = 0
    failed = 0
    for name, fn in tests:
        try:
            fn()
            passed += 1
            print(f"  ✓ {name}")
        except Exception as e:
            failed += 1
            print(f"  ✗ {name}: {e}")

    print(f"\n{'='*50}")
    print(f"INTEGRATION TEST RESULTS: {passed}/{passed+failed} passed")
    if failed == 0:
        print("ALL SYSTEMS INTEGRATED AND OPERATIONAL.")
    else:
        print(f"{failed} FAILURE(S) REQUIRING ATTENTION.")
    print(f"{'='*50}")
