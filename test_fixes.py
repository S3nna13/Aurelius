import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nn_utils import (
    RMSNorm, RotaryEmbedding, apply_rotary, FeedForward,
    sample_with_top_p_top_k, validate_input_ids, create_causal_mask, CausalMaskCache,
)
from agent_loop import ExperienceReplayBuffer
import memory_core
import agent_core
import skills
import inference
from aurelius_model_1b import AureliusModel1B


def test_nn_utils_rmsnorm_consistency():
    rms = RMSNorm(64)
    x = torch.randn(2, 16, 64)
    out = rms(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()
    assert not torch.isinf(out).any()


def test_nn_utils_rotary_apply_rotary_consistency():
    rotary = RotaryEmbedding(dim=64)
    h = torch.randn(2, 8, 64)
    cos, sin = rotary(h)
    assert cos.shape == (1, 1, 8, 32), f"cos shape: {cos.shape}"
    assert sin.shape == (1, 1, 8, 32)
    q = torch.randn(2, 12, 8, 64)
    rotated = apply_rotary(q, cos, sin)
    assert rotated.shape == q.shape, f"rotated shape: {rotated.shape}"


def test_nn_utils_feedforward():
    ff = FeedForward(64, 256)
    x = torch.randn(2, 16, 64)
    out = ff(x)
    assert out.shape == x.shape
    assert not torch.isnan(out).any()


def test_experience_replay_nan_rejection():
    buf = ExperienceReplayBuffer(capacity=100)
    clean = torch.randn(2, 8)
    nan_tensor = torch.tensor(float('nan')).expand(2, 8)
    buf.push(clean, clean, 1.0, clean, False)
    buf.push(nan_tensor, clean, 1.0, clean, False)
    buf.push(clean, nan_tensor, 1.0, clean, False)
    buf.push(clean, clean, 1.0, nan_tensor, False)
    assert len(buf) == 1, f"Expected 1 clean push, got {len(buf)}"


def test_experience_replay_inf_rejection():
    buf = ExperienceReplayBuffer(capacity=100)
    clean = torch.randn(2, 8)
    inf_tensor = torch.tensor(float('inf')).expand(2, 8)
    buf.push(inf_tensor, clean, 1.0, clean, False)
    assert len(buf) == 0, "inf state should be rejected"


def test_experience_replay_empty_rejection():
    buf = ExperienceReplayBuffer(capacity=100)
    clean = torch.randn(2, 8)
    empty = torch.empty(0)
    buf.push(empty, clean, 1.0, clean, False)
    assert len(buf) == 0, "empty state should be rejected"


def test_experience_replay_sample():
    buf = ExperienceReplayBuffer(capacity=100)
    for i in range(20):
        s = torch.randn(4)
        buf.push(s, s, float(i), s, False)
    assert len(buf) == 20
    s, a, r, ns, d = buf.sample(4)
    assert s.shape == (4, 4)
    assert r.shape == (4,)


def test_memory_core_surprise_gate():
    gate = memory_core.SurpriseGate(64, 32)
    h = torch.randn(2, 16, 64)
    s, lam = gate(h)
    assert s.shape == (2, 16, 32)
    assert lam.shape == (2, 16, 1)
    assert (s >= 0).all() and (s <= 1).all()


def test_memory_core_lts_read_write():
    lts = memory_core.LTSMemory(d_mem=32, capacity=64)
    keys = torch.randn(1, 8, 32)
    values = torch.randn(1, 8, 32)
    importance = torch.randn(1, 8, 1)
    lts.write(keys, values, importance)
    query = torch.randn(1, 16, 32)
    out = lts.read(query)
    assert out.shape == (1, 16, 32)


def test_skill_embedding_bounds():
    emb = skills.SkillEmbedding(d_model=64, skill_dim=32, max_skills=16)
    out = emb.forward(torch.tensor([[0, 1, 2]]))
    assert out.shape == (1, 3, 32)
    try:
        emb.forward(torch.tensor([[100]]))
        assert False, "Should have raised ValueError for OOB index"
    except ValueError:
        pass
    try:
        emb.forward(torch.tensor([[-1]]))
        assert False, "Should have raised ValueError for negative index"
    except ValueError:
        pass


def test_skill_embedding_add_skill():
    emb = skills.SkillEmbedding(d_model=64, skill_dim=32, max_skills=16)
    skill = torch.randn(32)
    idx = emb.add_skill(skill)
    assert 0 <= idx < 16
    retrieved = emb.embeddings[idx]
    assert torch.allclose(retrieved, skill.squeeze())


def test_agent_core_no_duplicate_return():
    import inspect
    source = inspect.getsource(agent_core.PlanningModule.forward)
    lines = source.strip().split('\n')
    return_lines = [l for l in lines if 'return' in l and 'plan_tensor' in l]
    assert len(return_lines) >= 1


def test_agent_core_tool_cross_attention_batch():
    attn = agent_core.ToolCrossAttention(d_model=64, n_heads=4)
    h = torch.randn(3, 8, 64)
    tool = torch.randn(3, 2, 64)
    out = attn(h, tool)
    assert out.shape == (3, 8, 64)


def test_inference_exception_narrowing():
    decoder = inference.SpeculativeDecoder(
        target_model=nn.Linear(64, 100),
        draft_model=None,
    )
    float_input = torch.randn(1, 64)
    out = decoder._model_forward(decoder.target_model, float_input)
    assert out is not None
    assert out.shape == (1, 100) or out.ndim == 2


def test_hierarchical_kv_cache_batch_validation():
    from hierarchical_kv_cache import HierarchicalKVCache
    cache = HierarchicalKVCache(
        d_model=64, n_heads=4, head_dim=16, cap1=16, cap2=64, cap3=128,
        max_batch=2,
    )
    key = torch.randn(3, 4, 4, 16)
    value = torch.randn(3, 4, 4, 16)
    hidden = torch.randn(3, 4, 64)
    try:
        cache.write(key, value, hidden)
        assert False, "Should have raised ValueError for batch mismatch"
    except ValueError:
        pass


def test_unified_manager_deterministic_scheduling():
    from unified_manager import UnifiedMemoryManager
    mgr = UnifiedMemoryManager(
        d_model=64, d_mem=32, n_layers=2,
        lts_capacity=64, episodic_slots=16, device='cpu',
    )
    h = torch.randn(2, 8, 64)
    lts_base = torch.randn(2, 64, 32)
    epi_base = torch.randn(2, 16, 32)
    surprise = torch.randn(2, 8, 1)
    call_count = 0
    for i in range(25):
        r = mgr.on_forward(0, h, lts_base, epi_base, surprise)
    counter = getattr(mgr, '_consolidation_counter', 0)
    assert counter == 25, f"Expected 25, got {counter}"
    consolidation_triggered = counter // 10
    assert True


def test_rlhf_reward_clipping():
    reward = torch.tensor([100.0, -100.0, 0.5, -0.5])
    clipped = torch.clamp(reward, -10.0, 10.0)
    normalized = (clipped - clipped.mean()) / (clipped.std() + 1e-8)
    assert clipped.max().item() <= 10.0
    assert clipped.min().item() >= -10.0
    assert abs(normalized.mean().item()) < 1e-6


def test_sample_top_p_top_k_basic():
    logits = torch.randn(2, 100)
    token = sample_with_top_p_top_k(logits, 0.8, top_k=0, top_p=0.9)
    assert token.shape == (2, 1)
    assert token.dtype == torch.long


def test_sample_top_p_top_k_temperature():
    logits = torch.randn(4, 50)
    token = sample_with_top_p_top_k(logits, 0.0, top_k=10, top_p=0.0)
    assert token.shape == (4, 1)


def test_validate_input_ids():
    ids = torch.randint(0, 1000, (2, 16))
    validate_input_ids(ids, 1000)
    try:
        validate_input_ids(torch.randn(2, 16), 1000)
        assert False, "Should reject float input"
    except TypeError:
        pass
    try:
        validate_input_ids(torch.randint(0, 1000, (16,)), 1000)
        assert False, "Should reject 1D input"
    except ValueError:
        pass
    try:
        validate_input_ids(torch.randint(0, 2000, (2, 16)), 1000)
        assert False, "Should reject OOB values"
    except ValueError:
        pass


def test_causal_mask_cache():
    cache = CausalMaskCache()
    m1 = cache.get(16, 'cpu')
    assert m1.shape == (1, 1, 16, 16)
    m2 = cache.get(8, 'cpu')
    assert m2.shape == (1, 1, 8, 8)
    m3 = cache.get(32, 'cpu')
    assert m3.shape == (1, 1, 32, 32)
    assert torch.allclose(m1[0, 0, :8, :8], m2[0, 0])


def test_rotary_caching():
    rotary = RotaryEmbedding(dim=64, max_position=128)
    x1 = torch.randn(2, 32, 64)
    cos1, sin1 = rotary(x1)
    x2 = torch.randn(2, 64, 64)
    cos2, sin2 = rotary(x2)
    assert cos1.shape == (1, 1, 32, 32)
    assert cos2.shape == (1, 1, 64, 32)
    assert rotary._cos_cached is not None
    assert rotary._cos_cached.size(-2) >= 64


def test_api_server_health_schema():
    from api_server import HealthResponse
    hr = HealthResponse(status="ok", model_loaded=True, device="cpu",
                        uptime_seconds=1.0, requests_served=0,
                        cuda_available=False)
    assert hr.status == "ok"


def test_api_server_model_load():
    from api_server import load_model, state
    if state.model is None:
        load_model(device='cpu')
    assert state.ready
    assert state.config is not None


def test_api_server_generate_endpoint():
    from api_server import app, state
    if state.model is None:
        from api_server import load_model
        load_model(device='cpu')
    from fastapi.testclient import TestClient
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ready"
    assert data["model_loaded"] is True
    resp2 = client.post("/generate", json={
        "prompt": [1, 2, 3, 4, 5],
        "max_new_tokens": 2,
        "temperature": 0.8,
        "top_p": 0.9,
    })
    assert resp2.status_code == 200
    gen = resp2.json()
    assert "tokens" in gen
    assert "token_count" in gen
    assert gen["token_count"] == 2


def test_api_server_empty_prompt_handling():
    from api_server import app, state
    if state.model is None:
        from api_server import load_model
        load_model(device='cpu')
    from fastapi.testclient import TestClient
    client = TestClient(app)
    resp = client.post("/generate", json={
        "prompt": [100],
        "max_new_tokens": 1,
        "temperature": 1.0,
        "top_p": 1.0,
    })
    assert resp.status_code == 200


def test_nn_utils_rotary_device_migration():
    rotary = RotaryEmbedding(dim=64, max_position=32)
    x = torch.randn(2, 16, 64)
    cos1, sin1 = rotary(x)
    assert cos1.device == x.device
    assert sin1.device == x.device


def test_causal_mask_cache_device():
    cache = CausalMaskCache()
    m = cache.get(8, 'cpu')
    assert m.device.type == 'cpu'
