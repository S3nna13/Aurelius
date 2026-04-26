"""Tests for warm-start helpers."""

import pytest
import torch
import torch.nn as nn

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.warm_start import (
    DepthGrowthScheduler,
    LayerDropout,
    WarmStartConfig,
    WarmStartInitializer,
    count_matchable_params,
    interpolation_warm_start,
    muggle_init,
    prefix_warm_start,
    warm_start_state,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

TEST_CONFIG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=4,
    n_kv_heads=2,
    head_dim=16,
    d_ff=128,
    vocab_size=256,
    max_seq_len=64,
    tie_embeddings=False,
)


def make_model() -> AureliusTransformer:
    torch.manual_seed(42)
    return AureliusTransformer(TEST_CONFIG)


# ---------------------------------------------------------------------------
# Original tests (preserved)
# ---------------------------------------------------------------------------


def test_warm_start_state_loads_matching_keys():
    target = {"a": torch.zeros(2), "b": torch.ones(3)}
    source = {"a": torch.full((2,), 5.0), "b": torch.full((3,), 2.0)}
    updated, report = warm_start_state(target, source)
    assert torch.equal(updated["a"], source["a"])
    assert report.loaded_keys == ("a", "b")


def test_warm_start_state_skips_missing_keys():
    updated, report = warm_start_state({"a": torch.zeros(2)}, {})
    assert torch.equal(updated["a"], torch.zeros(2))
    assert report.missing_keys == ("a",)


def test_warm_start_state_skips_shape_mismatch():
    updated, report = warm_start_state({"a": torch.zeros(2)}, {"a": torch.zeros(3)})
    assert updated["a"].shape == (2,)
    assert report.shape_mismatch_keys == ("a",)


def test_interpolation_warm_start_blends_tensors():
    target = torch.zeros(2)
    source = torch.ones(2)
    blended = interpolation_warm_start(target, source, alpha=0.25)
    assert torch.allclose(blended, torch.full((2,), 0.25))


def test_prefix_warm_start_copies_matching_prefix():
    target = torch.zeros(5)
    source = torch.tensor([1.0, 2.0, 3.0])
    result = prefix_warm_start(target, source)
    assert torch.allclose(result[:3], source)
    assert torch.allclose(result[3:], torch.zeros(2))


def test_interpolation_warm_start_rejects_bad_alpha():
    with pytest.raises(ValueError):
        interpolation_warm_start(torch.zeros(1), torch.zeros(1), alpha=1.5)


def test_prefix_warm_start_rejects_bad_dim():
    with pytest.raises(ValueError):
        prefix_warm_start(torch.zeros(2, 2), torch.zeros(2, 2), dim=2)


# ---------------------------------------------------------------------------
# New tests: WarmStartInitializer, LayerDropout, DepthGrowthScheduler,
#            muggle_init, count_matchable_params
# ---------------------------------------------------------------------------


def test_interpolate_weights_blends_params():
    """After alpha=1.0 interpolation, target params equal source params."""
    model = make_model()
    source_sd = {k: torch.ones_like(v) for k, v in model.named_parameters()}
    init = WarmStartInitializer(WarmStartConfig(strategy="interpolate", interpolation_alpha=1.0))
    init.interpolate_weights(model, source_sd, alpha=1.0)
    for name, param in model.named_parameters():
        if name in source_sd and source_sd[name].shape == param.shape:
            assert torch.allclose(param, source_sd[name]), f"Param {name} not blended"


def test_interpolate_skips_shape_mismatch():
    """A param with wrong shape in source is NOT overwritten in target."""
    model = make_model()
    first_name, first_param = next(model.named_parameters())
    original = first_param.detach().clone()

    source_sd = {first_name: torch.ones(1, 2, 3)}
    init = WarmStartInitializer()
    init.interpolate_weights(model, source_sd, alpha=1.0)

    current = dict(model.named_parameters())[first_name]
    assert torch.allclose(current, original), "Shape-mismatched param was modified"


def test_stack_layers_covers_all_target_layers():
    """All target layers are initialized from source (not left at default random)."""
    source_model = make_model()
    target_model = make_model()

    with torch.no_grad():
        for p in source_model.parameters():
            p.fill_(1.0)

    source_sd = dict(source_model.named_parameters())
    init = WarmStartInitializer()
    init.stack_layers(target_model, source_sd, target_n_layers=TEST_CONFIG.n_layers)

    for name, param in target_model.named_parameters():
        if name.startswith("layers."):
            assert torch.allclose(param, torch.ones_like(param)), (
                f"Layer param {name} not initialized from source"
            )


def test_layer_dropout_skips_during_train():
    """With drop_prob=1.0 in training mode, output equals input (layer skipped)."""
    inner = nn.Linear(8, 8)
    ld = LayerDropout(inner, drop_prob=1.0)
    ld.train()

    x = torch.randn(2, 8)
    out = ld(x)
    assert torch.equal(out, x), "LayerDropout should return x unchanged when skipped"


def test_layer_dropout_never_skips_eval():
    """With drop_prob=1.0 in eval mode, the layer IS applied (output differs from input)."""
    torch.manual_seed(0)
    inner = nn.Linear(8, 8)
    nn.init.normal_(inner.weight)
    nn.init.zeros_(inner.bias)

    ld = LayerDropout(inner, drop_prob=1.0)
    ld.eval()

    x = torch.randn(2, 8)
    out = ld(x)
    assert not torch.equal(out, x), "LayerDropout should NOT skip in eval mode"


def test_depth_growth_scheduler_anneals():
    """After anneal_steps/2, drop_prob is approximately initial_prob/2."""
    inner = nn.Identity()
    modules = [LayerDropout(inner, drop_prob=0.0) for _ in range(3)]
    initial = 0.5
    steps = 10000
    scheduler = DepthGrowthScheduler(modules, initial_prob=initial, anneal_steps=steps)

    current_prob = scheduler.step(steps // 2)
    assert abs(current_prob - initial / 2) < 1e-6, f"Expected ~{initial / 2}, got {current_prob}"
    for m in modules:
        assert abs(m.drop_prob - initial / 2) < 1e-6


def test_depth_growth_scheduler_reaches_zero():
    """At step >= anneal_steps, drop_prob == 0."""
    inner = nn.Identity()
    modules = [LayerDropout(inner, drop_prob=0.5)]
    scheduler = DepthGrowthScheduler(modules, initial_prob=0.5, anneal_steps=1000)

    prob = scheduler.step(1000)
    assert prob == 0.0
    assert modules[0].drop_prob == 0.0

    prob = scheduler.step(9999)
    assert prob == 0.0


def test_muggle_init_modifies_embed():
    """muggle_init changes the embed weight norm."""
    config = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
        tie_embeddings=False,
    )
    torch.manual_seed(7)
    model = AureliusTransformer(config)

    before_norm = model.embed.weight.norm().item()
    muggle_init(model)
    after_norm = model.embed.weight.norm().item()

    assert abs(before_norm - after_norm) > 1e-6, "muggle_init did not change embed weight norm"


def test_count_matchable_params_returns_correct_counts():
    """Test count_matchable_params with a small model and crafted source dict."""
    config = AureliusConfig(
        n_layers=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        head_dim=16,
        d_ff=128,
        vocab_size=256,
        max_seq_len=64,
        tie_embeddings=False,
    )
    model = AureliusTransformer(config)
    target_params = dict(model.named_parameters())

    target_names = list(target_params.keys())
    n_half = len(target_names) // 2

    source_sd: dict = {}
    for name in target_names[:n_half]:
        source_sd[name] = torch.zeros_like(target_params[name])
    mismatch_name = target_names[n_half]
    source_sd[mismatch_name] = torch.zeros(1, 2, 3)
    source_sd["__source_only__"] = torch.zeros(5)

    counts = count_matchable_params(model, source_sd)

    assert counts["n_matched"] == n_half
    assert counts["n_shape_mismatch"] == 1
    assert counts["n_source_only"] == 1
    expected_target_only = len(target_names) - n_half - 1
    assert counts["n_target_only"] == expected_target_only


def test_warm_start_apply_scratch():
    """strategy='scratch' leaves model weights unchanged."""
    model = make_model()
    original_params = {k: v.detach().clone() for k, v in model.named_parameters()}

    config = WarmStartConfig(strategy="scratch")
    init = WarmStartInitializer(config)
    init.apply(model)

    for name, param in model.named_parameters():
        assert torch.equal(param, original_params[name]), f"Scratch strategy modified param {name}"
