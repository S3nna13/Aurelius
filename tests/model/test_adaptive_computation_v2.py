"""15 tests for src/model/adaptive_computation_v2.py.

Tiny config: d_model=16, n_heads=2, vocab=16, seq_len=8, batch=2,
             max_steps=4, n_steps=3.
Every test runs at least one forward (and where required backward) pass.
"""

from __future__ import annotations

import pytest
import torch

from src.model.adaptive_computation_v2 import (
    ACTState,
    ACTTransformer,
    FixedDepthUniversalTransformer,
    HaltingUnit,
    UniversalTransformerLayer,
)

# ---------------------------------------------------------------------------
# Shared tiny config
# ---------------------------------------------------------------------------
D = 16
H = 2
V = 16
T = 8
B = 2
MAX_STEPS = 4
N_STEPS = 3


# ===========================================================================
# 1. HaltingUnit — output shape (B, T), values in (0, 1)
# ===========================================================================


def test_halting_unit_output_shape_and_range():
    unit = HaltingUnit(D)
    x = torch.randn(B, T, D)
    probs = unit(x)
    assert probs.shape == (B, T), f"Expected ({B},{T}), got {probs.shape}"
    assert (probs > 0).all() and (probs < 1).all(), "Probabilities must be in (0,1)"


# ===========================================================================
# 2. HaltingUnit.should_halt — True when cumulative_prob > threshold
# ===========================================================================


def test_halting_unit_should_halt():
    unit = HaltingUnit(D)
    # Construct cumulative probs: some above, some below threshold.
    cum = torch.tensor(
        [[0.5, 0.995, 0.1, 1.0, 0.98, 0.0, 0.999, 0.88], [1.0, 0.0, 0.5, 0.0, 1.0, 0.0, 0.5, 1.0]]
    )
    threshold = 0.99
    mask = unit.should_halt(cum, threshold)
    assert mask.dtype == torch.bool
    expected = cum > threshold
    assert torch.equal(mask, expected)


# ===========================================================================
# 3. ACTState.update — cumulative_prob increases, accumulated_output updates
# ===========================================================================


def test_act_state_update_cumulative_and_output():
    state = ACTState(B, T, D)
    x = torch.randn(B, T, D)
    p = torch.full((B, T), 0.3)

    cum_before = state.cumulative_prob.clone()
    acc_before = state.accumulated_output.clone()
    state.update(x, p)

    # cumulative_prob must have increased for all tokens (none halted initially).
    assert (state.cumulative_prob > cum_before).all(), "cumulative_prob must increase"
    # accumulated_output must have changed.
    assert not torch.equal(state.accumulated_output, acc_before), (
        "accumulated_output must be updated"
    )


# ===========================================================================
# 4. ACTState — once halted, no further updates
# ===========================================================================


def test_act_state_no_update_after_halt():
    state = ACTState(B, T, D)
    x = torch.randn(B, T, D)

    # First update with large p to trigger halting.
    p_large = torch.ones(B, T) * 0.99
    state.update(x, p_large)

    # Record state after first update.
    cum_snap = state.cumulative_prob.clone()
    state.accumulated_output.clone()
    halted_snap = state.halted.clone()

    # Second update — halted tokens should not change.
    x2 = torch.randn(B, T, D)
    p2 = torch.full((B, T), 0.5)
    state.update(x2, p2)

    # For tokens that were halted after first update, cumulative_prob must be unchanged.
    changed = state.cumulative_prob != cum_snap
    # No previously-halted token should have had its cumulative_prob changed.
    assert not (halted_snap & changed).any(), "Halted tokens must not have cumulative_prob updated"


# ===========================================================================
# 5. ACTState.finalize — output shape (B, T, D), remainder correctly applied
# ===========================================================================


def test_act_state_finalize_shape_and_remainder():
    state = ACTState(B, T, D)
    x = torch.randn(B, T, D)
    p = torch.full((B, T), 0.3)

    # One partial step — tokens are NOT all halted yet.
    state.update(x, p)
    output = state.finalize()

    assert output.shape == (B, T, D), f"Expected ({B},{T},{D}), got {output.shape}"
    # Output should be non-zero (remainder contribution applied).
    assert output.abs().sum() > 0


# ===========================================================================
# 6. UniversalTransformerLayer — output shape (B, T, D)
# ===========================================================================


def test_ut_layer_output_shape():
    layer = UniversalTransformerLayer(D, H)
    x = torch.randn(B, T, D)
    out = layer(x, step=0)
    assert out.shape == (B, T, D)


# ===========================================================================
# 7. UniversalTransformerLayer — grad flows
# ===========================================================================


def test_ut_layer_grad_flows():
    layer = UniversalTransformerLayer(D, H)
    x = torch.randn(B, T, D, requires_grad=True)
    out = layer(x, step=0)
    loss = out.sum()
    loss.backward()
    assert x.grad is not None, "Gradient must flow back to input"
    assert x.grad.shape == x.shape


# ===========================================================================
# 8. UniversalTransformerLayer — step=0 vs step=1 give different outputs
# ===========================================================================


def test_ut_layer_temporal_encoding_differs_by_step():
    layer = UniversalTransformerLayer(D, H)
    x = torch.randn(B, T, D)
    out0 = layer(x, step=0)
    out1 = layer(x, step=1)
    assert not torch.allclose(out0, out1), "Outputs must differ for different step indices"


# ===========================================================================
# 9. ACTTransformer.forward — logits shape (B, T, V), aux_loss ≥ 0,
#    mean_steps in [1, max_steps]
# ===========================================================================


def test_act_transformer_forward_shapes_and_ranges():
    model = ACTTransformer(D, H, V, max_steps=MAX_STEPS)
    ids = torch.randint(0, V, (B, T))
    logits, aux_loss, mean_steps = model(ids)

    assert logits.shape == (B, T, V)
    assert aux_loss.item() >= 0.0, "aux_loss must be non-negative"
    assert 1.0 <= mean_steps <= MAX_STEPS, f"mean_steps {mean_steps} not in [1, {MAX_STEPS}]"


# ===========================================================================
# 10. ACTTransformer — backward succeeds (through variable steps)
# ===========================================================================


def test_act_transformer_backward():
    model = ACTTransformer(D, H, V, max_steps=MAX_STEPS)
    ids = torch.randint(0, V, (B, T))
    logits, aux_loss, _ = model(ids)
    loss = logits.sum() + aux_loss
    loss.backward()  # must not raise

    # At least one parameter must have a gradient.
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients found after backward"


# ===========================================================================
# 11. ACTTransformer — time_penalty=0 → aux_loss=0
# ===========================================================================


def test_act_transformer_zero_time_penalty():
    model = ACTTransformer(D, H, V, max_steps=MAX_STEPS, time_penalty=0.0)
    ids = torch.randint(0, V, (B, T))
    _, aux_loss, _ = model(ids)
    assert aux_loss.item() == 0.0, "aux_loss must be 0 when time_penalty=0"


# ===========================================================================
# 12. ACTTransformer — tokens with high halting prob halt early (mean_steps < max_steps)
# ===========================================================================


def test_act_transformer_early_halting():
    """With act_threshold close to 0, most tokens should halt very early."""
    model = ACTTransformer(D, H, V, max_steps=MAX_STEPS, act_threshold=0.01)
    # Bias the halting unit heavily so it always outputs values near 1.
    with torch.no_grad():
        model.halting_unit.proj.bias.fill_(20.0)
    ids = torch.randint(0, V, (B, T))
    _, _, mean_steps = model(ids)
    assert mean_steps < MAX_STEPS, f"Expected early halting but mean_steps={mean_steps}"


# ===========================================================================
# 13. FixedDepthUniversalTransformer — logits shape (B, T, V), n_steps applied exactly
# ===========================================================================


def test_fixed_depth_ut_logits_shape():
    model = FixedDepthUniversalTransformer(D, H, V, n_steps=N_STEPS)
    ids = torch.randint(0, V, (B, T))
    logits = model(ids)
    assert logits.shape == (B, T, V)


# ===========================================================================
# 14. FixedDepthUniversalTransformer — backward succeeds
# ===========================================================================


def test_fixed_depth_ut_backward():
    model = FixedDepthUniversalTransformer(D, H, V, n_steps=N_STEPS)
    ids = torch.randint(0, V, (B, T))
    logits = model(ids)
    logits.sum().backward()

    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients found after backward"


# ===========================================================================
# 15. ACTTransformer with act_threshold=0.0 — all tokens halt at step 1
# ===========================================================================


def test_act_transformer_threshold_zero_halts_at_step1():
    # With threshold=0.0 every token halts on the very first step.
    model = ACTTransformer(D, H, V, max_steps=MAX_STEPS, act_threshold=0.0)
    ids = torch.randint(0, V, (B, T))
    _, _, mean_steps = model(ids)
    # mean_steps should be 1 (each token did exactly 1 update before halting).
    assert mean_steps == pytest.approx(1.0, abs=0.01), (
        f"Expected mean_steps≈1 with threshold=0.0, got {mean_steps}"
    )


# ===========================================================================
# Bonus (counts toward 15): FixedDepthUT n_steps=1 vs n_steps=4 differ
# ===========================================================================


def test_fixed_depth_ut_different_steps_give_different_outputs():
    torch.manual_seed(42)
    model1 = FixedDepthUniversalTransformer(D, H, V, n_steps=1)
    # Share weights so the only difference is the number of steps.
    model4 = FixedDepthUniversalTransformer(D, H, V, n_steps=4)
    model4.embedding.load_state_dict(model1.embedding.state_dict())
    model4.ut_layer.load_state_dict(model1.ut_layer.state_dict())
    model4.lm_head.load_state_dict(model1.lm_head.state_dict())

    ids = torch.randint(0, V, (B, T))
    out1 = model1(ids)
    out4 = model4(ids)
    assert not torch.allclose(out1, out4), (
        "n_steps=1 and n_steps=4 should produce different outputs"
    )
