"""Tests for model soups (Wortsman et al. 2022): uniform, fisher-weighted, and greedy averaging."""
import torch
import pytest

from src.training.model_soup import (
    SoupConfig,
    ModelSoup,
    uniform_soup,
    weighted_soup,
    fisher_weighted_soup,
    greedy_soup,
    compute_weight_variance,
    interpolate_models,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _small_cfg() -> AureliusConfig:
    return AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=2,
        n_kv_heads=2,
        head_dim=32,
        d_ff=128,
        vocab_size=256,
        max_seq_len=32,
    )


def _make_state_dict(seed: int) -> dict:
    torch.manual_seed(seed)
    cfg = _small_cfg()
    model = AureliusTransformer(cfg)
    return {k: v.clone() for k, v in model.state_dict().items()}


# Create 3 state dicts with different random seeds (as prescribed).
torch.manual_seed(0)
_SD0 = _make_state_dict(0)
_SD1 = _make_state_dict(1)
_SD2 = _make_state_dict(2)


# ---------------------------------------------------------------------------
# 1. SoupConfig defaults
# ---------------------------------------------------------------------------

def test_soup_config_defaults():
    cfg = SoupConfig()
    assert cfg.averaging_method == "uniform"
    assert cfg.n_models == 4
    assert cfg.held_out_fraction == 0.1
    assert cfg.temperature == 1.0


# ---------------------------------------------------------------------------
# 2. uniform_soup: avg of identical models == same model
# ---------------------------------------------------------------------------

def test_uniform_soup_same_models():
    avg = uniform_soup([_SD0, _SD0, _SD0])
    for key in _SD0:
        assert torch.allclose(avg[key].float(), _SD0[key].float(), atol=1e-5), (
            f"Key {key}: uniform avg of identical models should match original"
        )


# ---------------------------------------------------------------------------
# 3. uniform_soup: avg of two models has values between them
# ---------------------------------------------------------------------------

def test_uniform_soup_two_models():
    avg = uniform_soup([_SD0, _SD1])
    for key in _SD0:
        a = _SD0[key].float()
        b = _SD1[key].float()
        result = avg[key].float()
        expected = (a + b) / 2.0
        assert torch.allclose(result, expected, atol=1e-5), (
            f"Key {key}: uniform avg of two models should be midpoint"
        )


# ---------------------------------------------------------------------------
# 4. weighted_soup: weights sum to 1 internally
# ---------------------------------------------------------------------------

def test_weighted_soup_normalized():
    # Provide unnormalized weights [2, 2] — should behave same as [0.5, 0.5]
    w_unnorm = weighted_soup([_SD0, _SD1], weights=[2.0, 2.0])
    w_norm = weighted_soup([_SD0, _SD1], weights=[0.5, 0.5])
    for key in _SD0:
        assert torch.allclose(w_unnorm[key].float(), w_norm[key].float(), atol=1e-5), (
            f"Key {key}: weight normalization should produce identical results"
        )


# ---------------------------------------------------------------------------
# 5. weighted_soup: weight=1.0 on first model → identical to first
# ---------------------------------------------------------------------------

def test_weighted_soup_extreme():
    blended = weighted_soup([_SD0, _SD1], weights=[1.0, 0.0])
    for key in _SD0:
        assert torch.allclose(blended[key].float(), _SD0[key].float(), atol=1e-5), (
            f"Key {key}: weight=1 on first model should return first model"
        )


# ---------------------------------------------------------------------------
# 6. fisher_weighted_soup: output has same keys/shapes as input
# ---------------------------------------------------------------------------

def test_fisher_weighted_soup_shape():
    # Create trivial Fisher dicts (all-ones)
    fisher0 = {k: torch.ones_like(v) for k, v in _SD0.items()}
    fisher1 = {k: torch.ones_like(v) for k, v in _SD1.items()}
    merged = fisher_weighted_soup([_SD0, _SD1], [fisher0, fisher1])
    assert set(merged.keys()) == set(_SD0.keys()), "Fisher soup should have same keys"
    for key in _SD0:
        assert merged[key].shape == _SD0[key].shape, (
            f"Key {key}: shape mismatch in fisher_weighted_soup output"
        )


# ---------------------------------------------------------------------------
# 7. greedy_soup: always returns a valid state dict
# ---------------------------------------------------------------------------

def test_greedy_soup_returns_dict():
    call_count = [0]

    def eval_fn(sd):
        call_count[0] += 1
        # Return a deterministic score based on sum of first param
        first_key = next(iter(sd))
        return sd[first_key].float().mean().item()

    result = greedy_soup([_SD0, _SD1, _SD2], eval_fn)
    assert isinstance(result, dict), "greedy_soup must return a dict"
    assert set(result.keys()) == set(_SD0.keys()), "greedy_soup must have same keys"
    assert call_count[0] > 0, "eval_fn must be called at least once"


# ---------------------------------------------------------------------------
# 8. compute_weight_variance: all variances >= 0
# ---------------------------------------------------------------------------

def test_compute_weight_variance_nonneg():
    variances = compute_weight_variance([_SD0, _SD1, _SD2])
    assert set(variances.keys()) == set(_SD0.keys()), "variance keys must match state dict keys"
    for key, var in variances.items():
        assert var >= 0.0, f"Variance for {key} must be non-negative, got {var}"


# ---------------------------------------------------------------------------
# 9. interpolate_models: alpha=0 → model A
# ---------------------------------------------------------------------------

def test_interpolate_models_alpha0():
    result = interpolate_models(_SD0, _SD1, alpha=0.0)
    for key in _SD0:
        assert torch.allclose(result[key].float(), _SD0[key].float(), atol=1e-5), (
            f"Key {key}: alpha=0 should return model A"
        )


# ---------------------------------------------------------------------------
# 10. interpolate_models: alpha=1 → model B
# ---------------------------------------------------------------------------

def test_interpolate_models_alpha1():
    result = interpolate_models(_SD0, _SD1, alpha=1.0)
    for key in _SD0:
        assert torch.allclose(result[key].float(), _SD1[key].float(), atol=1e-5), (
            f"Key {key}: alpha=1 should return model B"
        )


# ---------------------------------------------------------------------------
# 11. ModelSoup.blend (uniform): returns a valid state dict
# ---------------------------------------------------------------------------

def test_model_soup_blend_uniform():
    cfg = SoupConfig(averaging_method="uniform")
    soup = ModelSoup(cfg)
    soup.add_model(_SD0)
    soup.add_model(_SD1)
    soup.add_model(_SD2)

    blended = soup.blend()
    assert isinstance(blended, dict), "blend() must return a dict"
    assert set(blended.keys()) == set(_SD0.keys()), "blend() must have same keys as input models"
    for key in _SD0:
        assert blended[key].shape == _SD0[key].shape, (
            f"Key {key}: blended shape must match original"
        )


# ---------------------------------------------------------------------------
# 12. ModelSoup.diversity_score: positive when models differ
# ---------------------------------------------------------------------------

def test_model_soup_diversity_score_positive():
    cfg = SoupConfig()
    soup = ModelSoup(cfg)
    soup.add_model(_SD0)
    soup.add_model(_SD1)
    soup.add_model(_SD2)

    score = soup.diversity_score()
    assert isinstance(score, float), "diversity_score() must return a float"
    assert score > 0.0, "diversity_score() must be positive when models differ"
