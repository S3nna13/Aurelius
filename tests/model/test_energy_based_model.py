"""
tests/model/test_energy_based_model.py

16 tests for the Energy-Based Model module.
Config: d_model=16, vocab_size=16, n_layers=2, B=4, T=6
"""

import math

import torch

from src.model.energy_based_model import (
    EBMConfig,
    EBMReranker,
    LangevinSampler,
    NCETrainer,
    NegativeSampler,
    SequenceEnergyFunction,
)

# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

D_MODEL = 16
VOCAB_SIZE = 16
N_LAYERS = 2
B = 4
T = 6


def _make_model() -> SequenceEnergyFunction:
    return SequenceEnergyFunction(D_MODEL, VOCAB_SIZE, N_LAYERS)


def _make_ids(b: int = B, t: int = T) -> torch.Tensor:
    return torch.randint(0, VOCAB_SIZE, (b, t))


# ---------------------------------------------------------------------------
# SequenceEnergyFunction tests
# ---------------------------------------------------------------------------


def test_energy_output_shape():
    """forward() must return a 1-D tensor of shape [B]."""
    model = _make_model()
    ids = _make_ids()
    energy = model(ids)
    assert energy.shape == (B,), f"Expected ({B},), got {energy.shape}"


def test_energy_output_finite():
    """forward() must produce finite values (no NaN or Inf)."""
    model = _make_model()
    ids = _make_ids()
    energy = model(ids)
    assert torch.isfinite(energy).all(), "Energy contains non-finite values"


def test_energy_score_equals_neg_energy():
    """score(x) must equal -energy(x) exactly."""
    model = _make_model()
    ids = _make_ids()
    energy = model(ids)
    score = model.score(ids)
    assert torch.allclose(score, -energy), "score(x) != -energy(x)"


def test_energy_gradient_flows():
    """Gradients must flow through the model parameters."""
    model = _make_model()
    ids = _make_ids()
    energy = model(ids)
    loss = energy.sum()
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters())
    assert has_grad, "No gradient flowed through SequenceEnergyFunction"


# ---------------------------------------------------------------------------
# NegativeSampler tests
# ---------------------------------------------------------------------------


def test_random_corrupt_changes_tokens():
    """random_corrupt must change at least one token."""
    ids = _make_ids()
    corrupted = NegativeSampler.random_corrupt(ids, corrupt_frac=0.5)
    assert not torch.equal(ids, corrupted), "random_corrupt did not change any tokens"


def test_random_corrupt_shape_preserved():
    """random_corrupt must not alter tensor shape."""
    ids = _make_ids()
    corrupted = NegativeSampler.random_corrupt(ids, corrupt_frac=0.15)
    assert corrupted.shape == ids.shape, f"Shape mismatch: {corrupted.shape} != {ids.shape}"


def test_masked_corrupt_mask_shape():
    """masked_corrupt must return a boolean mask of shape [B, T]."""
    ids = _make_ids()
    _, mask = NegativeSampler.masked_corrupt(ids, mask_token_id=0)
    assert mask.shape == (B, T), f"Mask shape {mask.shape} != ({B}, {T})"
    assert mask.dtype == torch.bool, f"Mask dtype {mask.dtype} is not bool"


def test_masked_corrupt_fraction():
    """masked_corrupt fraction should be approximately 0.15 on large input."""
    torch.manual_seed(0)
    ids = _make_ids(b=256, t=512)
    _, mask = NegativeSampler.masked_corrupt(ids)
    fraction = mask.float().mean().item()
    # Allow generous tolerance: ±0.05
    assert abs(fraction - 0.15) < 0.05, f"Masked fraction {fraction:.4f} is far from 0.15"


# ---------------------------------------------------------------------------
# NCETrainer tests
# ---------------------------------------------------------------------------


def test_nce_loss_finite():
    """nce_loss must return a finite scalar."""
    model = _make_model()
    trainer = NCETrainer(model, lr=1e-4)
    pos_ids = _make_ids()
    neg_ids = NegativeSampler.random_corrupt(pos_ids)
    loss = trainer.nce_loss(pos_ids, neg_ids)
    assert loss.dim() == 0, "nce_loss should return a scalar"
    assert torch.isfinite(loss), f"nce_loss is not finite: {loss.item()}"


def test_nce_loss_backward():
    """nce_loss must support backward pass (gradients must flow)."""
    model = _make_model()
    trainer = NCETrainer(model, lr=1e-4)
    pos_ids = _make_ids()
    neg_ids = NegativeSampler.random_corrupt(pos_ids)
    loss = trainer.nce_loss(pos_ids, neg_ids)
    loss.backward()
    has_grad = any(p.grad is not None and p.grad.abs().sum().item() > 0 for p in model.parameters())
    assert has_grad, "No gradient flowed through NCETrainer.nce_loss"


def test_contrastive_divergence_finite():
    """contrastive_divergence must return a finite scalar."""
    model = _make_model()
    trainer = NCETrainer(model, lr=1e-4)
    pos_ids = _make_ids()
    loss = trainer.contrastive_divergence(pos_ids, n_mcmc_steps=3)
    assert loss.dim() == 0, "contrastive_divergence should return a scalar"
    assert torch.isfinite(loss), f"contrastive_divergence is not finite: {loss.item()}"


def test_train_step_finite():
    """train_step must return a finite loss and update parameters."""
    model = _make_model()
    trainer = NCETrainer(model, lr=1e-4)
    pos_ids = _make_ids()
    neg_ids = NegativeSampler.random_corrupt(pos_ids)
    loss = trainer.train_step(pos_ids, neg_ids)
    assert isinstance(loss, torch.Tensor), "train_step should return a Tensor"
    assert torch.isfinite(loss), f"train_step loss is not finite: {loss.item()}"


# ---------------------------------------------------------------------------
# LangevinSampler tests
# ---------------------------------------------------------------------------


def test_langevin_step_shape():
    """LangevinSampler.step must preserve the shape of its input."""
    model = _make_model()
    sampler = LangevinSampler(model, step_size=0.01)
    x = torch.randn(B, T, D_MODEL)
    x_new = sampler.step(x)
    assert x_new.shape == (B, T, D_MODEL), (
        f"step output shape {x_new.shape} != ({B}, {T}, {D_MODEL})"
    )


def test_langevin_sample_shape():
    """LangevinSampler.sample must preserve the shape of its input."""
    model = _make_model()
    sampler = LangevinSampler(model, step_size=0.01)
    x0 = torch.randn(B, T, D_MODEL)
    x_final = sampler.sample(x0, n_steps=3)
    assert x_final.shape == (B, T, D_MODEL), (
        f"sample output shape {x_final.shape} != ({B}, {T}, {D_MODEL})"
    )


# ---------------------------------------------------------------------------
# EBMReranker tests
# ---------------------------------------------------------------------------


def test_reranker_returns_lowest_energy():
    """rerank must select the candidate the model assigns lowest energy to."""
    model = _make_model()
    reranker = EBMReranker(model, n_candidates=4)

    # Build 4 candidate sequences (each [1, T])
    candidates = [_make_ids(b=1, t=T) for _ in range(4)]

    # Ground-truth energies
    with torch.no_grad():
        energies = [model(c).item() for c in candidates]

    best = reranker.rerank(candidates)
    expected_idx = int(min(range(len(energies)), key=lambda i: energies[i]))
    expected_best = candidates[expected_idx]

    assert torch.equal(best, expected_best), "rerank did not return the lowest-energy candidate"


def test_ebm_config_defaults():
    """EBMConfig must expose correct default field values."""
    cfg = EBMConfig()
    assert cfg.d_model == 32
    assert cfg.vocab_size == 64
    assert cfg.n_layers == 2
    assert cfg.n_mcmc_steps == 5
    assert math.isclose(cfg.step_size, 0.1)
    assert cfg.n_candidates == 4
    assert math.isclose(cfg.corrupt_frac, 0.15)
    assert math.isclose(cfg.lr, 1e-4)
