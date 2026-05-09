import torch
import torch.nn as nn
from src.alignment.praxis.config import PRAXISConfig
from src.alignment.praxis.expert_safety_affinity import ExpertSafetyAffinity

class FakeTopKRouter(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.gate = nn.Linear(d_model, n_experts, bias=False)

class FakeFFN(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.router = FakeTopKRouter(d_model, n_experts)

class FakeTransformerBlock(nn.Module):
    def __init__(self, d_model, n_experts):
        super().__init__()
        self.ffn = FakeFFN(d_model, n_experts)

def make_fake_moe_layers(n_layers=2, d_model=8, n_experts=4):
    return nn.ModuleList([FakeTransformerBlock(d_model, n_experts) for _ in range(n_layers)])

def test_esa_loss_is_scalar_nonneg():
    cfg = PRAXISConfig(d_model=8, safety_experts=[0, 1], alpha_esa=0.01, tau_safety=0.5)
    moe_layers = make_fake_moe_layers(n_layers=2, d_model=8, n_experts=4)
    esa = ExpertSafetyAffinity(moe_layers, cfg)
    hidden = torch.randn(2, 6, 8)
    const_scores = torch.tensor([0.2, 0.8])   # first is unsafe
    loss = esa.compute(hidden, const_scores)
    assert loss.shape == (), f"expected scalar: {loss.shape}"
    assert loss.item() >= 0.0, f"ESA loss should be non-negative: {loss.item()}"

def test_esa_all_safe_gives_zero():
    cfg = PRAXISConfig(d_model=8, safety_experts=[0, 1], alpha_esa=0.01, tau_safety=0.5)
    moe_layers = make_fake_moe_layers(n_layers=2, d_model=8, n_experts=4)
    esa = ExpertSafetyAffinity(moe_layers, cfg)
    hidden = torch.randn(2, 6, 8)
    const_scores = torch.ones(2)   # all safe
    loss = esa.compute(hidden, const_scores)
    assert loss.item() == 0.0, "all-safe sequences → zero ESA loss"