import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recursive_mas import (
    RecursiveLink,
    RecursiveAgentWrapper,
    RecursiveMASConfig,
    InnerOuterOptimizer,
)


class DummyAgent(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(64, 128),
            nn.SiLU(),
            nn.Linear(128, 64),
        )

    def forward(self, x, tool_descs=None, full_cycle=False):
        return {'hidden': self.net(x)}


def test_recursive_link_2d():
    link = RecursiveLink(d_model=64, d_latent=32)
    h = torch.randn(4, 64)
    out = link(h)
    assert out.shape == (4, 64), f"2d out: {out.shape}"
    assert torch.isfinite(out).all()


def test_recursive_link_with_h_new():
    link = RecursiveLink(d_model=64, d_latent=32)
    h_prev = torch.randn(4, 64)
    h_new = torch.randn(4, 64)
    out = link(h_prev, h_new)
    assert out.shape == (4, 64)
    assert torch.isfinite(out).all()


def test_recursive_link_grads():
    link = RecursiveLink(d_model=64, d_latent=32)
    h = torch.randn(4, 64, requires_grad=True)
    out = link(h)
    loss = out.sum()
    loss.backward()
    assert h.grad is not None
    assert not torch.isnan(h.grad).any()
    for p in link.parameters():
        assert p.grad is not None
        assert not torch.isnan(p.grad).any()


def test_recursive_link_reset():
    link = RecursiveLink(d_model=64, d_latent=32)
    link.reset_parameters()
    for p in link.parameters():
        assert torch.isfinite(p).all()


def test_recursive_wrapper_shapes():
    wrapper = RecursiveAgentWrapper(
        d_model=64, d_latent=32,
        n_agents=2, agent_factory=DummyAgent,
        n_rounds=3,
    )
    h = torch.randn(2, 8, 64)
    out = wrapper(h)
    assert 'latent' in out
    assert out['latent'].shape == (2, 1, 64)
    assert out['value'].shape == (2,)
    assert out['confidence'].shape == (2, 1)
    assert out['n_rounds'] == 3
    assert out['n_agents'] == 2


def test_recursive_wrapper_return_rounds():
    wrapper = RecursiveAgentWrapper(
        d_model=64, d_latent=32,
        n_agents=2, agent_factory=DummyAgent,
        n_rounds=2,
    )
    h = torch.randn(1, 4, 64)
    out = wrapper(h, return_all_rounds=True)
    assert 'rounds' in out
    assert len(out['rounds']) == 4
    for r in out['rounds']:
        assert 'round' in r
        assert 'agent' in r
        assert 'latent' in r
        assert 'confidence' in r


def test_recursive_wrapper_grads():
    wrapper = RecursiveAgentWrapper(
        d_model=64, d_latent=32,
        n_agents=2, agent_factory=DummyAgent,
        n_rounds=2,
    )
    h = torch.randn(2, 8, 64, requires_grad=True)
    out = wrapper(h)
    loss = out['value'].mean() - out['confidence'].mean()
    loss.backward()
    assert h.grad is not None
    for name, p in wrapper.named_parameters():
        assert p.grad is not None, f"{name} missing grad"
        assert not torch.isnan(p.grad).any(), f"{name} has NaN grad"


def test_recursive_wrapper_single_agent():
    wrapper = RecursiveAgentWrapper(
        d_model=64, d_latent=32,
        n_agents=1, agent_factory=DummyAgent,
        n_rounds=1,
    )
    h = torch.randn(2, 8, 64)
    out = wrapper(h)
    assert out['latent'].shape == (2, 1, 64)
    assert out['n_rounds'] == 1
    assert out['n_agents'] == 1


def test_inner_outer_shapes():
    wrapper = RecursiveAgentWrapper(
        d_model=64, d_latent=32,
        n_agents=2, agent_factory=DummyAgent,
        n_rounds=2,
    )
    optim = InnerOuterOptimizer(wrapper, inner_lr=1e-3, outer_lr=1e-4)
    h = torch.randn(2, 4, 64)
    targets = {'value_labels': torch.randn(2)}
    task_embed = torch.randn(2, 32)
    n_rounds = wrapper.n_rounds * wrapper.n_agents
    rewards = torch.randn(2, n_rounds)
    metrics = optim.train_step(h, targets, task_embed, rewards)
    assert 'inner_loss' in metrics
    assert 'outer_loss' in metrics
    assert 'value_loss' in metrics
    assert 'confidence' in metrics
    assert 'avg_advantage' in metrics


def test_inner_outer_convergence():
    wrapper = RecursiveAgentWrapper(
        d_model=32, d_latent=16,
        n_agents=2,
        agent_factory=lambda: nn.Sequential(
            nn.Linear(32, 64), nn.SiLU(), nn.Linear(64, 32),
        ),
        n_rounds=2,
    )
    optim = InnerOuterOptimizer(wrapper, inner_lr=1e-2, outer_lr=1e-3)
    h = torch.randn(4, 2, 32)
    targets = {'value_labels': torch.randn(4)}
    task_embed = torch.randn(4, 16)
    rewards = torch.ones(4)
    losses = []
    for _ in range(5):
        m = optim.train_step(h, targets, task_embed, rewards)
        losses.append(m['inner_loss'])
    assert losses[-1] <= losses[0] + 0.1


def test_recursive_mas_config():
    cfg = RecursiveMASConfig(d_model=512, d_latent=128, n_agents=4, n_rounds=6)
    assert cfg.d_model == 512
    assert cfg.d_latent == 128
    assert cfg.n_agents == 4
    assert cfg.n_rounds == 6


def test_wrapper_plain_sequential_agent():
    wrapper = RecursiveAgentWrapper(
        d_model=16, d_latent=8,
        n_agents=2,
        agent_factory=lambda: nn.Sequential(
            nn.Linear(16, 32), nn.SiLU(), nn.Linear(32, 16),
        ),
        n_rounds=2,
    )
    h = torch.randn(2, 4, 16)
    out = wrapper(h)
    assert out['latent'].shape == (2, 1, 16)
    assert out['value'].shape == (2,)
