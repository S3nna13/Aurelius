import torch
import sys
sys.path.insert(0, '/Users/christienantonio/aurelius')
from brain_layer import *

B, D = 2, 64

def test_rmsnorm_shape():
    norm = RMSNorm(D)
    x = torch.randn(B, 8, D)
    out = norm(x)
    assert out.shape == (B, 8, D)

def test_rmsnorm_2d():
    norm = RMSNorm(D)
    x = torch.randn(B, D)
    out = norm(x)
    assert out.shape == (B, D)

def test_input_encoder_shape():
    enc = InputEncoder(d_brain=128, d_model=256)
    text = torch.randn(2, 10, 256)
    out = enc(text)
    assert out.shape == (2, 128)

def test_working_memory_shape():
    wm = WorkingMemory(d_brain=128, n_slots=32)
    s = torch.randn(2, 128)
    out = wm(s)
    assert out.shape == (2, 128)
    wm.reset()
    assert wm.slots.sum().item() == 0.0

def test_reasoning_core_shape():
    rc = ReasoningCore(d_brain=D)
    wm = torch.randn(2, D)
    goal = torch.randn(2, D)
    consensus, conf = rc(wm, goal, ltm=[], n_steps=5)
    assert consensus.shape == (2, D)
    assert conf.shape == (2, 1)

def test_planner_shape():
    p = Planner(d_brain=D, max_subtasks=8)
    goal = torch.randn(2, D)
    wm = torch.randn(2, D)
    subtasks, deps = p(goal, wm)
    assert subtasks.shape == (2, 8, D)
    assert deps.shape == (2, 8, 8)

def test_verifier_shape():
    v = Verifier(d_brain=D, n_errors=5)
    step = torch.randn(2, D)
    wm = torch.randn(2, D)
    err_prob, err_types, conf, fix = v(step, wm)
    assert err_prob.shape == (2, 1)
    assert err_types.shape == (2, 5)
    assert conf.shape == (2, 1)
    assert fix.shape == (2, D)

def test_critic_shape():
    c = Critic(d_brain=D)
    state = torch.randn(2, D)
    action = torch.randn(2, D)
    score, suggestion = c(state, action)
    assert score.shape == (2, 1)
    assert suggestion.shape == (2, D)

def test_uncertainty_estimator_shape():
    ue = UncertaintyEstimator(d_brain=D)
    h = torch.randn(2, 8, D)
    epi, alea, ent = ue(h)
    assert epi.shape == (2, 8, 1)
    assert alea.shape == (2, 8, 1)
    assert ent.shape == (2, 8, 1)

def test_full_brain_forward():
    brain = NeuralBrainLayer(d_brain=D, d_model=128, n_slots=16, max_steps=5)
    text = torch.randn(1, 8, 128)
    result = brain(text)
    assert result['output'].shape == (1, D)
    assert 'confidence' in result

def test_gated_residual():
    gr = GatedResidual(d_model=D)
    h = torch.randn(B, D)
    res = torch.randn(B, D)
    out = gr(h, res)
    assert out.shape == (B, D)
