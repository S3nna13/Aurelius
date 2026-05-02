import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import memory_core
import agent_core
import skills
import agent_loop
import speculative_decoding
try:
    import moe_memory
except ImportError:
    import archive.moe_memory as moe_memory
try:
    import ntm_memory
except ImportError:
    import archive.ntm_memory as ntm_memory
import hierarchical_kv_cache
import paged_optimizer
import fp8_allreduce
import rlhf_lora
import mobile_inference
import skills_registry
import agent_registry
import api_registry
import tool_schema_registry

B = 2
T = 8
D = 64


# ─── memory_core.py ───────────────────────────────────────────────────────────

def test_memory_core_forward_shape():
    m = memory_core.AurelianMemoryCore(d_model=D, d_mem=D, episodic_slots=16, lts_capacity=32)
    h = torch.randn(B, T, D)
    out = m(h)
    assert out.shape == (B, T, D)


def test_memory_core_backward_non_nan():
    m = memory_core.AurelianMemoryCore(d_model=D, d_mem=D, episodic_slots=16, lts_capacity=32)
    h = torch.randn(B, T, D)
    out = m(h)
    loss = out.sum()
    loss.backward()
    has_grad = False
    for n, p in m.named_parameters():
        if p.grad is not None:
            has_grad = True
            assert not torch.isnan(p.grad).any(), f"{n} has NaN"
            assert not torch.isinf(p.grad).any(), f"{n} has inf"
    assert has_grad


def test_memory_core_return_mem_state():
    m = memory_core.AurelianMemoryCore(d_model=D, d_mem=D, episodic_slots=16, lts_capacity=32)
    h = torch.randn(B, T, D)
    out, state = m(h, return_mem_state=True)
    assert 'surprise' in state
    assert 'lambda' in state
    assert 'mem_read' in state
    assert state['surprise'].shape == (B, T, D)
    assert state['lambda'].shape == (B, T, 1)


def test_surprise_gate_non_zero_grads():
    gate = memory_core.SurpriseGate(d_model=D, d_mem=D)
    h = torch.randn(B, T, D, requires_grad=True)
    s, lam = gate(h)
    s.sum().backward()
    assert h.grad is not None
    assert h.grad.abs().sum() > 0


def test_surprise_gate_shapes():
    gate = memory_core.SurpriseGate(d_model=D, d_mem=D // 2)
    h = torch.randn(B, T, D)
    s, lam = gate(h)
    assert s.shape == (B, T, D // 2)
    assert lam.shape == (B, T, 1)


def test_graph_consolidator_forward():
    gc = memory_core.GraphConsolidator(d_mem=D)
    slots = torch.randn(B, 8, D)
    out = gc(slots)
    assert out.shape == (B, 8, D)


def test_bigru_encoder_empty():
    enc = memory_core.BiGRUSlotEncoder(d_mem=D)
    slots = torch.randn(B, 0, D)
    out = enc(slots)
    assert out.shape == (B, 0, D)


def test_bigru_encoder_shape():
    enc = memory_core.BiGRUSlotEncoder(d_mem=D)
    slots = torch.randn(B, 8, D)
    out = enc(slots)
    assert out.shape == (B, 8, D)


# ─── agent_core.py ────────────────────────────────────────────────────────────

def test_tool_embedding_no_args():
    te = agent_core.ToolEmbedding(d_model=D, max_tools=16)
    out = te()
    assert out.shape == (1, D)


def test_tool_embedding_with_ids():
    te = agent_core.ToolEmbedding(d_model=D, max_tools=16)
    ids = torch.randint(0, 16, (B, 4))
    out = te(tool_ids=ids)
    assert out.shape == (B, 4, D)


def test_tool_embedding_with_descs():
    te = agent_core.ToolEmbedding(d_model=D, max_tools=16)
    descs = torch.randn(B, 4, 64)
    out = te(tool_descs=descs)
    assert out.shape == (B, 4, D)


def test_tool_former_adapter_shape():
    adapter = agent_core.ToolFormerAdapter(d_model=D, n_heads=4, n_known_tools=16)
    h = torch.randn(B, T, D)
    out, call = adapter(h)
    assert out.shape == (B, T, D)
    assert call is None


def test_tool_former_adapter_return_call():
    adapter = agent_core.ToolFormerAdapter(d_model=D, n_heads=4, n_known_tools=16)
    h = torch.randn(B, T, D)
    out, call = adapter(h, return_call=True)
    assert call is not None
    logits, presence, params = call
    assert logits.shape == (B, 16)
    assert presence.shape == (B, 8)
    assert params.shape == (B, 64)


def test_tool_call_head_shapes():
    head = agent_core.ToolCallHead(d_model=D, n_known_tools=16, max_params=8)
    h = torch.randn(B, T, D)
    logits, presence, params = head(h)
    assert logits.shape == (B, 16)
    assert presence.shape == (B, 8)
    assert params.shape == (B, 64)


def test_tool_call_head_grads():
    head = agent_core.ToolCallHead(d_model=D, n_known_tools=16, max_params=8)
    h = torch.randn(B, T, D, requires_grad=True)
    logits, presence, params = head(h)
    logits.sum().backward()
    assert h.grad is not None
    assert not torch.isnan(h.grad).any()


def test_cross_attention_shape():
    attn = agent_core.ToolCrossAttention(d_model=D, n_heads=4)
    h = torch.randn(B, T, D)
    tool = torch.randn(B, 2, D)
    out = attn(h, tool)
    assert out.shape == (B, T, D)


def test_tool_result_integrator():
    integ = agent_core.ToolResultIntegrator(d_model=D)
    h = torch.randn(B, T, D)
    result = torch.randn(B, T, D)
    out = integ(h, result)
    assert out.shape == (B, T, D)


def test_critic_head_shape():
    critic = agent_core.CriticHead(d_model=D)
    state = torch.randn(B, T, D)
    action = torch.randn(B, D)
    score, suggestion = critic(state, action)
    assert score.shape == (B,)
    assert suggestion.shape == (B, D)


def test_value_head_shape():
    vh = agent_core.ValueHead(d_model=D)
    h = torch.randn(B, T, D)
    out = vh(h)
    assert out.shape == (B,)


def test_planning_module_mcts_nodes():
    planner = agent_core.PlanningModule(d_model=D, n_simulations=2, max_depth=2)
    root_state = torch.randn(1, D)
    root = agent_core.MCTSNode(root_state)
    for i in range(3):
        emb = planner.action_proposer(root.state)
        child = agent_core.MCTSNode(emb + root.state, parent=root, action_idx=i)
        root.children.append(child)
    assert len(root.children) == 3
    best = max(root.children, key=lambda c: c.ucb_score())
    assert best is not None


def test_mcts_node_value():
    node = agent_core.MCTSNode(torch.randn(D))
    assert node.value() == 0.0
    node.visits = 5
    node.value_sum = 3.0
    assert node.value() == 0.6


def test_mcts_node_ucb():
    root = agent_core.MCTSNode(torch.randn(D))
    root.visits = 10
    child = agent_core.MCTSNode(torch.randn(D), parent=root, action_idx=0)
    child.visits = 2
    child.value_sum = 1.0
    child.prior = 0.5
    score = child.ucb_score(c_puct=1.4)
    assert score > 0


def test_mcts_node_ucb_inf():
    root = agent_core.MCTSNode(torch.randn(D))
    root.visits = 10
    child = agent_core.MCTSNode(torch.randn(D), parent=root, action_idx=0)
    child.prior = 0.5
    score = child.ucb_score(c_puct=1.4)
    assert score == float('inf')


# ─── skills.py ────────────────────────────────────────────────────────────────

def test_skill_embedding_shape():
    emb = skills.SkillEmbedding(d_model=D, skill_dim=D, max_skills=16)
    ids = torch.randint(0, 16, (B,))
    out = emb(ids)
    assert out.shape == (B, D)


def test_skill_embedding_all():
    emb = skills.SkillEmbedding(d_model=D, skill_dim=D, max_skills=16)
    out = emb()
    assert out.shape == (16, D)


def test_skill_embedding_add_skill():
    emb = skills.SkillEmbedding(d_model=D, skill_dim=D, max_skills=16)
    new = torch.randn(D)
    idx = emb.add_skill(new)
    retrieved = emb.embeddings[idx]
    assert torch.allclose(retrieved, new.squeeze())


def test_skill_embedding_named():
    emb = skills.SkillEmbedding(d_model=D, skill_dim=D, max_skills=16)
    names = torch.randn(B, 64)
    out = emb.get_named_skills(names)
    assert out.shape == (B, D)


def test_skill_acquisition_extract():
    acq = skills.SkillAcquisition(d_model=D, skill_dim=D)
    traj = torch.randn(B, T, D)
    skill = acq.extract_skill(traj)
    assert skill.shape == (B, D)


def test_skill_acquisition_update():
    acq = skills.SkillAcquisition(d_model=D, skill_dim=D)
    old = torch.randn(D)
    new = torch.randn(D)
    updated = acq.update_embedding(old, new)
    assert updated.shape == (D,)


def test_skill_controller_shape():
    ctrl = skills.SkillController(d_model=D, skill_dim=D)
    h = torch.randn(B, T, D)
    skill = torch.randn(B, D)
    out = ctrl(h, skill)
    assert out.shape == (B, T, D)


def test_skill_controller_grads():
    ctrl = skills.SkillController(d_model=D, skill_dim=D)
    h = torch.randn(B, T, D, requires_grad=True)
    skill = torch.randn(B, D)
    out = ctrl(h, skill)
    out.sum().backward()
    assert h.grad is not None
    assert not torch.isnan(h.grad).any()


def test_skill_execution_adapter_shape():
    adapt = skills.SkillExecutionAdapter(d_model=D, skill_dim=D, n_heads=4)
    h = torch.randn(B, 1, D)
    skill = torch.randn(B, D)
    out = adapt(h, skill)
    assert out.shape == (B, 1, D)


def test_skill_registry_add_skill():
    reg = skills.SkillRegistry(d_model=D, skill_dim=D, max_skills=16, n_top_k=4)
    skill = torch.randn(D)
    idx = reg.embedding.add_skill(skill, idx=5)
    assert idx == 5
    assert torch.allclose(reg.embedding.embeddings[5], skill)


def test_skill_registry_update_skill():
    reg = skills.SkillRegistry(d_model=D, skill_dim=D, max_skills=16, n_top_k=4)
    old = reg.embedding.embeddings[0].clone()
    new_skill = reg.acquisition.update_embedding(old, torch.randn(D))
    assert new_skill.shape == (D,)


def test_skill_registry_get_top():
    reg = skills.SkillRegistry(d_model=D, skill_dim=D, max_skills=16, n_top_k=4)
    reg.skill_success_rate[0] = 0.9
    reg.skill_usage_count[0] = 10
    reg.skill_success_rate[1] = 0.1
    reg.skill_usage_count[1] = 5
    indices, values = reg.get_top_skills(k=2)
    assert indices.shape == (2,)
    assert values.shape == (2,)


def test_skill_library_compose():
    lib = skills.SkillLibrary(d_model=D, skill_dim=D, max_skills=16)
    a = torch.randn(D)
    b = torch.randn(D)
    composed = lib.compose_skills(a, b)
    assert composed.shape == (D,)


# ─── skills_registry.py ────────────────────────────────────────────────────────

def test_skills_registry_contract_forward():
    reg = skills_registry.SkillRegistryContract(d_model=D, skill_dim=D, max_skills=16, n_top_k=4)
    reg.verify_contract(batch=B, time=T)


def test_skills_registry_lookup():
    entry = skills_registry.lookup("skill_registry")
    assert entry is not None
    assert entry.live is True
    assert "SkillRegistry" in entry.name


def test_skills_registry_lookup_missing():
    entry = skills_registry.lookup("nonexistent")
    assert entry is None


def test_skills_registry_get_registry():
    reg = skills_registry.get_registry()
    assert len(reg) >= 6
    for name, entry in reg.items():
        assert entry.live is True
        assert entry.contract != ""
        assert entry.path != ""


# ─── agent_registry.py ────────────────────────────────────────────────────────

def test_agent_registry_tool_embedding_contract():
    c = agent_registry.ToolEmbeddingContract(d_model=D, max_tools=16)
    c.verify_contract(batch=B, time=T)


def test_agent_registry_tool_former_adapter_contract():
    c = agent_registry.ToolFormerAdapterContract(d_model=D, n_heads=4, n_known_tools=16)
    c.verify_contract(batch=B, time=T)


def test_agent_registry_planning_module_contract():
    c = agent_registry.PlanningModuleContract(d_model=D, n_simulations=2, max_depth=2)
    c.verify_contract(batch=B, time=T)


def test_agent_registry_agent_loop_controller_contract():
    c = agent_registry.AgentLoopControllerContract(d_model=D, n_heads=4, d_mem=D, n_known_tools=8, n_simulations=2)
    c.verify_contract(batch=B, time=T)


def test_agent_registry_memory_bridge_contract():
    c = agent_registry.AgentMemoryBridgeContract(d_model=D, d_mem=D, episodic_slots=16)
    c.verify_contract(batch=B, time=T)


def test_agent_registry_critic_head_contract():
    c = agent_registry.CriticHeadContract(d_model=D)
    c.verify_contract(batch=B, time=T)


def test_agent_registry_value_head_contract():
    c = agent_registry.ValueHeadContract(d_model=D)
    c.verify_contract(batch=B, time=T)


def test_agent_registry_lookup():
    entry = agent_registry.lookup("tool_former_adapter")
    assert entry is not None
    assert entry.live is True


def test_agent_registry_lookup_missing():
    assert agent_registry.lookup("nope") is None


def test_agent_registry_get_registry():
    reg = agent_registry.get_registry()
    assert len(reg) >= 10
    for entry in reg.values():
        assert entry.live is True
        assert entry.contract != ""
        assert entry.path != ""


# ─── api_registry.py ──────────────────────────────────────────────────────────

def test_api_registry_lookup():
    entry = api_registry.lookup("train")
    assert entry is not None
    assert entry.live is True


def test_api_registry_lookup_missing():
    assert api_registry.lookup("nope") is None


def test_api_registry_get_registry():
    reg = api_registry.get_registry()
    assert len(reg) >= 10
    for entry in reg.values():
        assert entry.live is True
        assert entry.contract != ""
        assert entry.path != ""


def test_api_registry_config_contract():
    c = api_registry.ConfigContract("config.yaml", "aurelius_150m")
    c.verify_contract()


# ─── tool_schema_registry.py ──────────────────────────────────────────────────

def test_tool_schema_registry_lookup():
    entry = tool_schema_registry.lookup("memory_core")
    assert entry is not None
    assert entry.live is True


def test_tool_schema_registry_lookup_missing():
    assert tool_schema_registry.lookup("nope") is None


def test_tool_schema_registry_get_registry():
    reg = tool_schema_registry.get_registry()
    assert len(reg) >= 10
    for entry in reg.values():
        assert entry.live is True
        assert entry.contract != ""
        assert entry.path != ""


def test_tool_schema_registry_verify_imports():
    results = tool_schema_registry.verify_imports()
    for name, ok in results.items():
        assert ok, f"import failed for {name}"


# ─── agent_loop.py ────────────────────────────────────────────────────────────

def test_agent_loop_observe():
    loop = agent_loop.AgentLoopController(d_model=D, n_heads=4, d_mem=D, n_known_tools=8, n_simulations=2)
    h = torch.randn(B, T, D)
    out = loop.observe(h)
    assert out.shape == (B, T, D)


def test_agent_loop_think_value_head():
    loop = agent_loop.AgentLoopController(d_model=D, n_heads=4, d_mem=D, n_known_tools=8, n_simulations=2)
    h = torch.randn(B, T, D)
    v = loop.value_head(h)
    assert v.shape == (B,)


def test_agent_memory_read():
    bridge = agent_loop.AgentMemoryBridge(d_model=D, d_mem=D, episodic_slots=16)
    agent_state = torch.randn(B, T, D)
    mem_slots = torch.randn(1, 16, D)
    out = bridge.read_from_memory(agent_state, mem_slots)
    assert out.shape == (B, T, D)


def test_agent_memory_grads():
    bridge = agent_loop.AgentMemoryBridge(d_model=D, d_mem=D, episodic_slots=16)
    agent_state = torch.randn(B, T, D, requires_grad=True)
    mem_slots = torch.randn(1, 16, D)
    out = bridge.read_from_memory(agent_state, mem_slots)
    out.sum().backward()
    assert agent_state.grad is not None
    assert not torch.isnan(agent_state.grad).any()


def test_context_manager():
    ctx = agent_loop.AgentContextManager(max_context=16, window_size=8)
    for i in range(20):
        ctx.add({'action': f'step_{i}'})
    recent = ctx.get_recent(4)
    assert len(recent) == 4
    summary = ctx.get_task_summary()
    assert 'Recent actions' in summary


def test_context_manager_empty():
    ctx = agent_loop.AgentContextManager(max_context=16, window_size=8)
    assert ctx.get_task_summary() == ''


def test_experience_replay_push_sample():
    buf = agent_loop.ExperienceReplayBuffer(capacity=16)
    for i in range(8):
        s = torch.randn(D)
        a = torch.randn(D)
        r = float(i)
        ns = torch.randn(D)
        buf.push(s, a, r, ns, done=(i == 7))
    assert len(buf) == 8
    states, actions, rewards, next_states, dones = buf.sample(2)
    assert states.shape == (2, D)
    assert actions.shape == (2, D)
    assert rewards.shape == (2,)
    assert next_states.shape == (2, D)
    assert dones.shape == (2,)


def test_agent_episode_dataclass():
    ep = agent_loop.AgentEpisode()
    assert ep.total_reward == 0.0
    assert len(ep.steps) == 0


# ─── speculative_decoding.py ──────────────────────────────────────────────────

def test_memory_context_projector_shape():
    proj = speculative_decoding.MemoryContextProjector(d_mem=D, d_draft=D)
    mem = torch.randn(B, D)
    out = proj(mem)
    assert out.shape == (B, D)


def test_memory_context_projector_grads():
    proj = speculative_decoding.MemoryContextProjector(d_mem=D, d_draft=D)
    mem = torch.randn(B, D, requires_grad=True)
    out = proj(mem)
    out.sum().backward()
    assert mem.grad is not None
    assert not torch.isnan(mem.grad).any()


def test_draft_model_forward_shape():
    V = 32
    model = speculative_decoding.MemoryAwareDraftModel(vocab_size=V, d_model=D, n_heads=4, n_layers=2, d_mem=D)
    ids = torch.randint(0, V, (B, T))
    mem = torch.randn(B, D)
    logits = model(ids, mem)
    assert logits.shape == (B, T, V)


def test_draft_model_forward_grads():
    V = 32
    model = speculative_decoding.MemoryAwareDraftModel(vocab_size=V, d_model=D, n_heads=4, n_layers=2, d_mem=D)
    ids = torch.randint(0, V, (B, T))
    mem = torch.randn(B, D, requires_grad=True)
    logits = model(ids, mem)
    logits.sum().backward()
    assert mem.grad is not None
    assert not torch.isnan(mem.grad).any()


def test_draft_model_generate_draft_shape():
    V = 32
    model = speculative_decoding.MemoryAwareDraftModel(vocab_size=V, d_model=D, n_heads=4, n_layers=2, d_mem=D)
    ids = torch.randint(0, V, (B, 4))
    mem = torch.randn(B, D)
    draft = model.generate_draft(ids, mem, gamma=3, temperature=0.8, top_k=10, top_p=0.9)
    assert draft.shape == (B, 3)


def test_draft_model_generate_greedy():
    V = 32
    model = speculative_decoding.MemoryAwareDraftModel(vocab_size=V, d_model=D, n_heads=4, n_layers=2, d_mem=D)
    ids = torch.randint(0, V, (1, 4))
    mem = torch.randn(1, D)
    draft = model.generate_draft(ids, mem, gamma=2, temperature=0.0, top_k=0, top_p=1.0)
    assert draft.shape == (1, 2)


def test_speculative_config_defaults():
    cfg = speculative_decoding.SpeculativeConfig()
    assert cfg.gamma == 5
    assert cfg.temperature == 0.8
    assert cfg.top_k == 50
    assert cfg.top_p == 0.9


class _MockTargetModel(nn.Module):
    def __init__(self, V, d):
        super().__init__()
        self.embed = nn.Embedding(V, d)
        self.head = nn.Linear(d, V)
    def forward(self, input_ids):
        return self.head(self.embed(input_ids))


def test_speculative_decoder_generate():
    V = 16; DS = 16
    draft = speculative_decoding.MemoryAwareDraftModel(vocab_size=V, d_model=DS, n_heads=2, n_layers=1, d_mem=DS)
    target = _MockTargetModel(V, DS)
    decoder = speculative_decoding.SpeculativeDecoder(speculative_decoding.SpeculativeConfig(gamma=2, temperature=0.8, top_k=5, top_p=0.9))
    ids = torch.randint(0, V, (1, 4))
    mem = torch.randn(1, DS)
    out = decoder.generate_with_speculation(ids, target, draft, mem, max_new_tokens=4, gamma=2, temperature=0.8, top_k=5, top_p=0.9)
    assert out.shape[0] == 1
    assert out.shape[1] >= 4


# ─── moe_memory.py ────────────────────────────────────────────────────────────

def test_moe_router_shape():
    router = moe_memory.MoEMemoryRouter(d_model=D, n_experts=4, top_k=2)
    x = torch.randn(B, T, D)
    w, idx = router(x)
    assert w.shape == (B, T, 2)
    assert idx.shape == (B, T, 2)


def test_moe_router_top_k_one():
    router = moe_memory.MoEMemoryRouter(d_model=D, n_experts=4, top_k=1)
    x = torch.randn(B, T, D)
    w, idx = router(x)
    assert w.shape == (B, T, 1)


def test_moe_memory_forward_shape():
    moe = moe_memory.MoELTSMemory(d_model=D, d_mem=D, n_experts=4, top_k=2, capacity=16)
    h = torch.randn(B, T, D)
    q = torch.randn(B, T, D)
    out, loss = moe(h, q)
    assert out.shape == (B, T, D)
    assert loss.shape == ()


def test_moe_load_balancing_loss():
    moe = moe_memory.MoELTSMemory(d_model=D, d_mem=D, n_experts=4, top_k=2, capacity=16)
    h = torch.randn(B, T, D)
    q = torch.randn(B, T, D)
    _, loss = moe(h, q)
    assert loss.item() >= 0.0
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)


def test_moe_memory_backward():
    moe = moe_memory.MoELTSMemory(d_model=D, d_mem=D, n_experts=4, top_k=2, capacity=16)
    h = torch.randn(B, T, D, requires_grad=True)
    q = torch.randn(B, T, D)
    out, loss = moe(h, q)
    (out.sum() + loss).backward()
    assert h.grad is not None
    assert not torch.isnan(h.grad).any()


def test_memory_expert_read():
    exp = moe_memory.MemoryExpert(d_mem=D, capacity=16)
    q = torch.randn(B, T, D)
    out = exp.read(q)
    assert out.shape == (B, T, D)


def test_memory_expert_forward():
    exp = moe_memory.MemoryExpert(d_mem=D, capacity=16)
    q = torch.randn(B, T, D)
    out = exp(q)
    assert out.shape == (B, T, D)


def test_memory_expert_consolidate():
    exp = moe_memory.MemoryExpert(d_mem=D, capacity=16)
    old = exp.memory.data.clone()
    exp.consolidate(lr=0.1)
    assert exp.memory.data.shape == old.shape


# ─── ntm_memory.py ────────────────────────────────────────────────────────────

def test_ntm_reset_shape():
    ntm = ntm_memory.NTMMemory(N=32, M=D)
    mem = ntm.reset(batch_size=B)
    assert mem.shape == (B, 32, D)


def test_ntm_read_weight():
    ntm = ntm_memory.NTMMemory(N=16, M=D)
    mem = ntm.reset(batch_size=B)
    rw = F.softmax(torch.randn(B, 16), dim=-1)
    read = ntm.read(mem, rw)
    assert read.shape == (B, D)


def test_ntm_write_update():
    ntm = ntm_memory.NTMMemory(N=16, M=D)
    mem = ntm.reset(batch_size=B)
    ww = F.softmax(torch.randn(B, 16), dim=-1)
    ev = torch.sigmoid(torch.randn(B, D))
    av = torch.randn(B, D)
    updated = ntm.write(mem, ww, ev, av)
    assert updated.shape == (B, 16, D)


def test_ntm_read_head_shape():
    head = ntm_memory.NTMReadHead(d_mem=D, N=32, shift_radius=1)
    q = torch.randn(B, D)
    k = torch.randn(B, D)
    mem = torch.randn(B, 32, D)
    pw = torch.zeros(B, 32)
    pw[:, 0] = 1.0
    w = head(q, k, mem, pw)
    assert w.shape == (B, 32)


def test_ntm_read_head_sharpening():
    head = ntm_memory.NTMReadHead(d_mem=D, N=16, shift_radius=1)
    q = torch.randn(B, D)
    k = torch.randn(B, D)
    mem = torch.randn(B, 16, D)
    pw = F.softmax(torch.randn(B, 16), dim=-1)
    w = head(q, k, mem, pw)
    assert w.shape == (B, 16)
    assert torch.allclose(w.sum(dim=-1), torch.ones(B))


def test_ntm_write_head_shape():
    head = ntm_memory.NTMWriteHead(d_mem=D, N=32, shift_radius=1)
    q = torch.randn(B, D)
    k = torch.randn(B, D)
    mem = torch.randn(B, 32, D)
    pw = torch.zeros(B, 32)
    pw[:, 0] = 1.0
    ww, ev, av = head(q, k, mem, pw)
    assert ww.shape == (B, 32)
    assert ev.shape == (B, D)
    assert av.shape == (B, D)


def test_ntm_controller_forward():
    ctrl = ntm_memory.NTMController(d_controller=D, d_mem=D, N=32, shift_radius=1)
    mem = torch.zeros(1, 32, D)
    prw = torch.zeros(1, 32)
    prw[:, 0] = 1.0
    pww = torch.zeros(1, 32)
    pww[:, 0] = 1.0
    co = torch.randn(1, D)
    read, state = ctrl(co, (mem, prw, pww))
    mem_u, new_rw, new_ww = state
    assert read.shape == (1, D)
    assert mem_u.shape == (1, 32, D)
    assert new_rw.shape == (1, 32)
    assert new_ww.shape == (1, 32)


def test_ntm_controller_grads():
    ctrl = ntm_memory.NTMController(d_controller=D, d_mem=D, N=32, shift_radius=1)
    mem = torch.zeros(1, 32, D)
    prw = torch.zeros(1, 32)
    prw[:, 0] = 1.0
    pww = torch.zeros(1, 32)
    pww[:, 0] = 1.0
    co = torch.randn(1, D, requires_grad=True)
    read, state = ctrl(co, (mem, prw, pww))
    read.sum().backward()
    assert co.grad is not None
    assert not torch.isnan(co.grad).any()


def test_damb_init_state():
    block = ntm_memory.DifferentiableMemoryAugmentedBlock(d_model=D, d_mem=D, N=16)
    state = block.init_state(batch_size=B)
    mem, rw, ww = state
    assert mem.shape == (B, 16, D)
    assert rw.shape == (B, 16)
    assert ww.shape == (B, 16)
    assert rw[:, 0].sum().item() == B


# ─── hierarchical_kv_cache.py ─────────────────────────────────────────────────

def test_hierarchical_cache_read_empty():
    cache = hierarchical_kv_cache.HierarchicalKVCache(d_model=D, n_heads=4, head_dim=16, cap1=8, cap2=16, cap3=32, max_batch=1)
    k, v, labels = cache.read()
    assert k.shape[2] == 0
    assert v.shape[2] == 0
    assert labels.shape[1] == 0


def test_hierarchical_cache_write_t1_read():
    cache = hierarchical_kv_cache.HierarchicalKVCache(d_model=D, n_heads=4, head_dim=16, cap1=8, cap2=16, cap3=32, max_batch=1)
    key = torch.randn(1, 4, 4, 16)
    value = torch.randn(1, 4, 4, 16)
    hidden = torch.randn(1, 4, 64)
    cache.write(key, value, hidden)
    k, v, labels = cache.read()
    assert k.shape[1] == 4
    assert k.shape[2] > 0
    assert labels.shape[1] > 0


def test_hierarchical_cache_write_cascade():
    cache = hierarchical_kv_cache.HierarchicalKVCache(d_model=D, n_heads=4, head_dim=16, cap1=4, cap2=8, cap3=16, max_batch=1)
    for _ in range(3):
        key = torch.randn(1, 4, 4, 16)
        value = torch.randn(1, 4, 4, 16)
        hidden = torch.randn(1, 4, 64)
        cache.write(key, value, hidden)
    k, v, labels = cache.read()
    assert k.shape[2] > 0


def test_hierarchical_cache_read_tier():
    cache = hierarchical_kv_cache.HierarchicalKVCache(d_model=D, n_heads=4, head_dim=16, cap1=4, cap2=8, cap3=16, max_batch=1)
    key = torch.randn(1, 4, 2, 16)
    value = torch.randn(1, 4, 2, 16)
    hidden = torch.randn(1, 2, 64)
    cache.write(key, value, hidden)
    k1, v1 = cache.read_tier(1)
    assert k1.shape[2] > 0


def test_hierarchical_cache_read_tier2_empty():
    cache = hierarchical_kv_cache.HierarchicalKVCache(d_model=D, n_heads=4, head_dim=16, cap1=4, cap2=8, cap3=16, max_batch=1)
    k2, v2 = cache.read_tier(2)
    assert k2.shape[2] == 0


def test_hierarchical_cache_reset():
    cache = hierarchical_kv_cache.HierarchicalKVCache(d_model=D, n_heads=4, head_dim=16, cap1=4, cap2=8, cap3=16, max_batch=1)
    key = torch.randn(1, 4, 2, 16)
    value = torch.randn(1, 4, 2, 16)
    hidden = torch.randn(1, 2, 64)
    cache.write(key, value, hidden)
    cache.reset()
    k, v, _ = cache.read()
    assert k.shape[2] == 0


def test_hierarchical_write_small_batch():
    cache = hierarchical_kv_cache.HierarchicalKVCache(d_model=D, n_heads=4, head_dim=16, cap1=4, cap2=8, cap3=16, max_batch=2)
    key = torch.randn(2, 4, 2, 16)
    value = torch.randn(2, 4, 2, 16)
    hidden = torch.randn(2, 2, 64)
    cache.write(key, value, hidden)
    k, v, labels = cache.read()
    assert k.shape[0] == 2


def test_multi_scale_attention_shape():
    msa = hierarchical_kv_cache.MultiScaleAttention(d_model=D, n_heads=4)
    x = torch.randn(B, T, D)
    k_cache = torch.randn(B, 4, 16, D // 4)
    v_cache = torch.randn(B, 4, 16, D // 4)
    labels = torch.zeros(B, 16, dtype=torch.long)
    out = msa(x, (k_cache, v_cache, labels), causal_mask=False)
    assert out.shape == (B, T, D)


def test_importance_scorer_shape():
    scorer = hierarchical_kv_cache.ImportanceScorer(d_model=D)
    k = torch.randn(B, 4, 8, D // 4)
    v = torch.randn(B, 4, 8, D // 4)
    h = torch.randn(B, 8, D)
    scores = scorer(k, v, h)
    assert scores.shape == (B, 8)


def test_importance_scorer_grads():
    scorer = hierarchical_kv_cache.ImportanceScorer(d_model=D)
    k = torch.randn(B, 4, 4, D // 4, requires_grad=True)
    v = torch.randn(B, 4, 4, D // 4)
    h = torch.randn(B, 4, D)
    scores = scorer(k, v, h)
    scores.sum().backward()
    assert k.grad is not None
    assert not torch.isnan(k.grad).any()


def test_eviction_policy_evict():
    policy = hierarchical_kv_cache.LearnedEvictionPolicy(d_model=D)
    k = torch.randn(1, 4, 8, 16)
    v = torch.randn(1, 4, 8, 16)
    scores = torch.randn(1, 8)
    keep_k, keep_v, keep_s, evict_k, evict_v, evict_s = policy.evict(k, v, scores, n_evict=3)
    assert keep_k.shape[2] == 5
    assert evict_k.shape[2] == 3
    assert keep_s.shape[1] == 5


# ─── paged_optimizer.py ───────────────────────────────────────────────────────

def test_paged_adamw_state_management():
    model = nn.Linear(4, 4)
    opt = paged_optimizer.PagedAdamW(model.parameters(), lr=0.01, gpu_budget=2)
    assert len(opt.param_groups) > 0


def test_paged_adamw_register():
    mgr = paged_optimizer.PagedOptimizerState(gpu_budget=2)
    mgr.register_param('p0', 0)
    mgr.register_param('p1', 0)
    t = torch.zeros(4)
    mgr.set_state('p0', t, t)
    state = mgr.get_state('p0')
    assert state is not None


def test_compressor_round_trip():
    state = {
        'test': {
            'exp_avg': torch.randn(4, 4),
            'exp_avg_sq': torch.randn(4, 4),
            'step': 3,
        }
    }
    comp = paged_optimizer.OptimizerStateCompressor.compress(state)
    assert comp['test']['exp_avg_sq'].dtype == torch.half
    decomp = paged_optimizer.OptimizerStateCompressor.decompress(comp)
    assert decomp['test']['exp_avg_sq'].dtype == torch.float
    assert decomp['test']['step'] == 3
    assert torch.allclose(state['test']['exp_avg_sq'].float(), decomp['test']['exp_avg_sq'].float(), rtol=1e-2)


def test_gradient_bucket_add_flush():
    bucket = paged_optimizer.GradientBucket(bucket_size_mb=1)
    g1 = torch.randn(4, 4)
    g2 = torch.randn(4, 4)
    bucket.add_grad('g1', g1)
    bucket.add_grad('g2', g2)
    buckets = bucket.flush()
    assert len(buckets) > 0
    bucket.reset()
    assert len(bucket._gradients) == 0


def test_gradient_bucket_accumulate():
    bucket = paged_optimizer.GradientBucket(bucket_size_mb=1)
    g = torch.randn(8, 8)
    bucket.add_grad('g', g)
    bucket.add_grad('g', g.clone())
    assert 'g' in bucket._gradients


# ─── fp8_allreduce.py ─────────────────────────────────────────────────────────

FP8_OK = torch.cuda.is_available()

def test_fp8_compressor_ratio():
    comp = fp8_allreduce.FP8Compressor()
    assert comp.get_compression_ratio() == 4.0


def test_fp8_compress_decompress_round_trip():
    if not FP8_OK: return
    comp = fp8_allreduce.FP8Compressor()
    t = torch.randn(4, 4, device='cuda')
    q, scale = comp.compress(t)
    dq = comp.decompress(q, scale)
    assert dq.shape == t.shape
    assert dq.dtype == torch.float32


def test_fp8_compress_decompress_zero():
    if not FP8_OK: return
    comp = fp8_allreduce.FP8Compressor()
    t = torch.zeros(4, 4, device='cuda')
    q, scale = comp.compress(t)
    assert scale.item() == 1.0
    dq = comp.decompress(q, scale)
    assert dq.abs().sum().item() == 0


def test_fp8_error_feedback_buffer():
    buf = fp8_allreduce.ErrorFeedbackBuffer()
    orig = torch.randn(4, 4)
    quant = torch.randn(4, 4)
    buf.push('test', orig, quant)
    err = buf.pop('test')
    assert err is not None
    assert err.shape == (4, 4)
    assert buf.pop('test') is None


def test_fp8_error_feedback_accumulate():
    buf = fp8_allreduce.ErrorFeedbackBuffer()
    g1 = torch.randn(4, 4)
    g2 = torch.randn(4, 4)
    buf.push('x', g1, torch.zeros_like(g1))
    buf.push('x', g2, torch.zeros_like(g2))
    err = buf.pop('x')
    assert torch.allclose(err, g1 + g2)


def test_fp8_error_feedback_clear():
    buf = fp8_allreduce.ErrorFeedbackBuffer()
    buf.push('a', torch.randn(4), torch.randn(4))
    buf.clear()
    assert buf.pop('a') is None


def test_fp8_communication_budget():
    budget = fp8_allreduce.CommunicationBudget()
    budget.record(1024)
    budget.record(2048)
    report = budget.report()
    assert 'bytes' in report
    budget.reset()
    assert budget._bytes == 0


def test_fp8_communication_budget_empty():
    budget = fp8_allreduce.CommunicationBudget()
    report = budget.report()
    assert '0 bytes' in report


# ─── rlhf_lora.py ─────────────────────────────────────────────────────────────

def test_lora_layer_forward_shape():
    lora = rlhf_lora.LoraLayer(d_in=D, d_out=D, r=8, alpha=16, dropout=0.1)
    x = torch.randn(B, T, D)
    out = lora(x)
    assert out.shape == (B, T, D)


def test_lora_layer_forward_grads():
    lora = rlhf_lora.LoraLayer(d_in=D, d_out=D, r=4, alpha=8, dropout=0.0)
    x = torch.randn(B, T, D, requires_grad=True)
    out = lora(x)
    out.sum().backward()
    assert lora.A.grad is not None
    assert lora.B.grad is not None
    assert not torch.isnan(lora.A.grad).any()
    assert not torch.isnan(lora.B.grad).any()


def test_lora_merge_into_weight():
    lora = rlhf_lora.LoraLayer(d_in=16, d_out=16, r=4, alpha=8, dropout=0.0)
    W = torch.randn(16, 16)
    merged = lora.merge_into_weight(W)
    assert merged.shape == (16, 16)
    delta = (lora.A @ lora.B) * (8.0 / 4.0)
    assert torch.allclose(merged, W + delta, atol=1e-5)


def test_lora_layer_dropout():
    lora = rlhf_lora.LoraLayer(d_in=D, d_out=D, r=8, alpha=16, dropout=0.5)
    lora.train()
    x = torch.randn(B, T, D)
    out1 = lora(x)
    out2 = lora(x)
    assert out1.shape == out2.shape


def test_lora_layer_reset():
    lora = rlhf_lora.LoraLayer(d_in=D, d_out=D, r=8, alpha=16, dropout=0.1)
    lora.reset()
    assert torch.allclose(lora.B, torch.zeros_like(lora.B))
    assert not torch.allclose(lora.A, torch.zeros_like(lora.A))


def test_lora_layer_eval():
    lora = rlhf_lora.LoraLayer(d_in=D, d_out=D, r=8, alpha=16, dropout=0.5)
    lora.eval()
    x = torch.randn(B, T, D)
    out = lora(x)
    assert out.shape == (B, T, D)


# ─── mobile_inference.py ──────────────────────────────────────────────────────

def test_mobile_quantizer_quantize_dequantize():
    q = mobile_inference.MobileQuantizer()
    t = torch.randn(4, 4)
    q_t, scale = q.quantize_tensor(t, bits=8)
    dq = q.dequantize_tensor(q_t, scale)
    assert q_t.dtype == torch.int8
    assert dq.shape == t.shape
    assert dq.dtype == torch.float32


def test_mobile_quantizer_zero():
    q = mobile_inference.MobileQuantizer()
    t = torch.zeros(4, 4)
    q_t, scale = q.quantize_tensor(t, bits=8)
    dq = q.dequantize_tensor(q_t, scale)
    assert dq.abs().sum().item() == 0


def test_mobile_quantizer_different_bits():
    q = mobile_inference.MobileQuantizer()
    t = torch.randn(4, 4)
    q4, s4 = q.quantize_tensor(t, bits=4)
    assert q4.dtype == torch.int8
    dq4 = q.dequantize_tensor(q4, s4)
    assert dq4.shape == (4, 4)


def test_pruned_head_forward_shape():
    head = mobile_inference.PrunedHead(d_model=D, vocab_size=32, rank=16)
    x = torch.randn(B, T, D)
    out = head(x)
    assert out.shape == (B, T, 32)


def test_pruned_head_grads():
    head = mobile_inference.PrunedHead(d_model=D, vocab_size=32, rank=16)
    x = torch.randn(B, T, D, requires_grad=True)
    out = head(x)
    out.sum().backward()
    assert head.U.grad is not None
    assert head.S.grad is not None
    assert head.Vh.grad is not None
    assert not torch.isnan(head.U.grad).any()


def test_pruned_head_prune():
    head = mobile_inference.PrunedHead(d_model=32, vocab_size=32, rank=8)
    W = torch.randn(32, 32)
    head.prune(W, rank=4)
    assert head.U.shape[1] == 4


def test_rms_norm_shape():
    norm = mobile_inference.RMSNorm(dim=D)
    x = torch.randn(B, T, D)
    out = norm(x)
    assert out.shape == (B, T, D)


def test_rms_norm_grads():
    norm = mobile_inference.RMSNorm(dim=D)
    x = torch.randn(B, T, D, requires_grad=True)
    out = norm(x)
    out.sum().backward()
    assert x.grad is not None
    assert not torch.isnan(x.grad).any()


def test_mobile_memory_manager_init():
    mgr = mobile_inference.MobileMemoryManager(d_model=D, device_buffer_size=8)
    assert mgr.device_buffer.shape == (8, D)
    assert mgr.mmap_size == 0


def test_inference_optimizer():
    optim = mobile_inference.InferenceOptimizer()
    model = nn.Sequential(nn.Linear(4, 4), nn.ReLU(), nn.Linear(4, 2))
    model.eval()
    compiled = optim.optimize(model, mode='reduce-overhead')
    assert compiled is not None


def test_inference_optimizer_fuse():
    optim = mobile_inference.InferenceOptimizer()
    model = nn.Sequential(nn.LayerNorm(4), nn.Linear(4, 4))
    fused = optim._fuse_norm_linear(model)
    x = torch.randn(2, 4)
    with torch.no_grad():
        out = fused(x)
    assert out.shape == (2, 4)
