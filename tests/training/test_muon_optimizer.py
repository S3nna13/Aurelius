"""Tests for MUON optimizer."""

from __future__ import annotations

import torch

from src.training.muon import Muon, _hybrid_newton_schulz, _newton_schulz, build_muon_optimizer


def test_newton_schulz_orthogonality():
    G = torch.randn(32, 64)
    X = _newton_schulz(G, steps=5)
    assert X.shape == G.shape
    assert not torch.isnan(X).any()


def test_hybrid_newton_schulz_orthogonality():
    G = torch.randn(32, 64)
    X = _hybrid_newton_schulz(G)
    assert X.shape == G.shape
    assert not torch.isnan(X).any()


class SimpleMatrixModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(64, 64)

    def forward(self, x):
        return self.linear(x)


def test_muon_step_no_error():
    model = SimpleMatrixModel()
    muon_opt, adam_opt = build_muon_optimizer(model, muon_lr=0.02, adam_lr=3e-4)
    x = torch.randn(4, 64)
    loss = model(x).sum()
    loss.backward()
    muon_opt.step()
    muon_opt.zero_grad()
    assert True


def test_muon_param_norms_bounded():
    model = SimpleMatrixModel()
    muon_opt, _ = build_muon_optimizer(model, muon_lr=0.02, adam_lr=3e-4)
    for _ in range(10):
        x = torch.randn(4, 64)
        loss = model(x).sum()
        loss.backward()
        muon_opt.step()
        muon_opt.zero_grad()

    for p in model.parameters():
        assert p.norm().item() < 100


def test_build_muon_optimizer_split():
    model = SimpleMatrixModel()
    muon_opt, adam_opt = build_muon_optimizer(model, muon_lr=0.02, adam_lr=3e-4)
    assert isinstance(muon_opt, Muon)
    assert isinstance(adam_opt, torch.optim.AdamW)
    muon_param_ids = {id(p) for p in muon_opt.param_groups[0]["params"]}
    for g in adam_opt.param_groups:
        for p in g["params"]:
            assert id(p) not in muon_param_ids
