"""Tests for DoRA (Weight-Decomposed Low-Rank Adaptation).

Import path: aurelius.training.dora
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
from aurelius.training.dora import DoRALinear, DoRAModel, _col_norms

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def simple_linear() -> nn.Linear:
    """A small nn.Linear for testing."""
    torch.manual_seed(42)
    linear = nn.Linear(16, 8, bias=True)
    return linear


@pytest.fixture()
def dora_from_linear(simple_linear) -> DoRALinear:
    """DoRALinear constructed from the simple_linear fixture."""
    return DoRALinear.from_linear(simple_linear, rank=4)


# ---------------------------------------------------------------------------
# Test 1: Output shape matches nn.Linear
# ---------------------------------------------------------------------------


def test_output_shape_matches_linear(simple_linear, dora_from_linear):
    """DoRALinear output shape must equal nn.Linear output shape."""
    torch.manual_seed(0)
    x = torch.randn(3, 16)
    expected = simple_linear(x)
    actual = dora_from_linear(x)
    assert actual.shape == expected.shape, f"Shape mismatch: {actual.shape} vs {expected.shape}"


# ---------------------------------------------------------------------------
# Test 2: At init (B=0), output equals original Linear
# ---------------------------------------------------------------------------


def test_output_equals_original_at_init(simple_linear, dora_from_linear):
    """With B=0, DoRALinear output must equal the original nn.Linear output."""
    torch.manual_seed(0)
    x = torch.randn(5, 16)
    with torch.no_grad():
        expected = simple_linear(x)
        actual = dora_from_linear(x)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 3: m initialized to column norms of weight_0
# ---------------------------------------------------------------------------


def test_m_initialized_to_col_norms(simple_linear, dora_from_linear):
    """m should equal the per-output-column norms of weight_0."""
    expected_norms = _col_norms(dora_from_linear.weight_0)  # (1, out)
    torch.testing.assert_close(dora_from_linear.m.data, expected_norms, rtol=1e-5, atol=1e-6)


# ---------------------------------------------------------------------------
# Test 4: A initialized non-zero, B initialized zero
# ---------------------------------------------------------------------------


def test_lora_A_nonzero_B_zero():
    """lora_A should be non-zero; lora_B should be all zeros at init."""
    dora = DoRALinear(in_features=16, out_features=8, rank=4)
    assert dora.lora_A.data.abs().sum().item() > 0, "lora_A should be non-zero"
    assert dora.lora_B.data.abs().sum().item() == 0, "lora_B should be zero at init"


# ---------------------------------------------------------------------------
# Test 5: Only m, A, B have requires_grad; weight_0 is frozen
# ---------------------------------------------------------------------------


def test_grad_flags(dora_from_linear):
    """weight_0 must be frozen; m, lora_A, lora_B must be trainable."""
    # weight_0 is a buffer, so it's not in named_parameters; confirm no grad
    assert not dora_from_linear.weight_0.requires_grad, "weight_0 must be frozen"

    # Trainable parameters
    assert dora_from_linear.m.requires_grad, "m must require grad"
    assert dora_from_linear.lora_A.requires_grad, "lora_A must require grad"
    assert dora_from_linear.lora_B.requires_grad, "lora_B must require grad"

    # bias is also trainable
    if dora_from_linear.bias is not None:
        assert dora_from_linear.bias.requires_grad, "bias must require grad"


# ---------------------------------------------------------------------------
# Test 6: Gradients flow through m and B after loss.backward()
# ---------------------------------------------------------------------------


def test_gradients_flow(dora_from_linear):
    """After backward, m and lora_B must have non-None gradients."""
    torch.manual_seed(1)
    x = torch.randn(4, 16)
    out = dora_from_linear(x)
    loss = out.sum()
    loss.backward()

    assert dora_from_linear.m.grad is not None, "m.grad should not be None"
    assert dora_from_linear.lora_B.grad is not None, "lora_B.grad should not be None"
    assert dora_from_linear.lora_A.grad is not None, "lora_A.grad should not be None"
    # weight_0 has no grad (it's a buffer)
    assert dora_from_linear.weight_0.grad is None, "weight_0 must not accumulate grad"


# ---------------------------------------------------------------------------
# Test 7: from_linear preserves weight and bias
# ---------------------------------------------------------------------------


def test_from_linear_preserves_weight_and_bias(simple_linear):
    """from_linear must copy weight and bias from the source Linear."""
    dora = DoRALinear.from_linear(simple_linear, rank=4)
    torch.testing.assert_close(dora.weight_0, simple_linear.weight.detach())
    torch.testing.assert_close(dora.bias.data, simple_linear.bias.detach())


# ---------------------------------------------------------------------------
# Test 8: merge() returns nn.Linear with correct output
# ---------------------------------------------------------------------------


def test_merge_returns_linear_with_correct_output(dora_from_linear):
    """Merged nn.Linear must produce the same output as DoRALinear."""
    torch.manual_seed(2)
    x = torch.randn(6, 16)
    with torch.no_grad():
        dora_out = dora_from_linear(x)
        merged = dora_from_linear.merge()
        merged_out = merged(x)
    torch.testing.assert_close(dora_out, merged_out, rtol=1e-4, atol=1e-5)
    assert isinstance(merged, nn.Linear), "merge() must return nn.Linear"


# ---------------------------------------------------------------------------
# Test 9: After one optimizer step, output changes
# ---------------------------------------------------------------------------


def test_output_changes_after_optimizer_step(dora_from_linear):
    """After one gradient step, the layer output must differ from initial."""
    torch.manual_seed(3)
    x = torch.randn(4, 16)
    with torch.no_grad():
        out_before = dora_from_linear(x).clone()

    optimizer = torch.optim.SGD(
        [dora_from_linear.m, dora_from_linear.lora_A, dora_from_linear.lora_B],
        lr=0.1,
    )
    out = dora_from_linear(x)
    loss = out.sum()
    loss.backward()
    optimizer.step()

    with torch.no_grad():
        out_after = dora_from_linear(x)

    assert not torch.allclose(out_before, out_after), "Output should change after an optimizer step"


# ---------------------------------------------------------------------------
# Test 10: DoRAModel replaces target layers
# ---------------------------------------------------------------------------


def test_doramodel_replaces_target_layers():
    """DoRAModel must swap the specified Linear layers for DoRALinear."""

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(8, 8)
            self.v_proj = nn.Linear(8, 8)
            self.fc = nn.Linear(8, 4)  # should NOT be replaced

        def forward(self, x):
            return self.fc(self.q_proj(x) + self.v_proj(x))

    model = TinyModel()
    dora_model = DoRAModel(model, target_modules=["q_proj", "v_proj"], rank=2)

    assert isinstance(dora_model.model.q_proj, DoRALinear), "q_proj must be DoRALinear"
    assert isinstance(dora_model.model.v_proj, DoRALinear), "v_proj must be DoRALinear"
    assert isinstance(dora_model.model.fc, nn.Linear) and not isinstance(
        dora_model.model.fc, DoRALinear
    ), "fc must remain plain nn.Linear"


# ---------------------------------------------------------------------------
# Test 11: DoRAModel.trainable_parameters excludes weight_0
# ---------------------------------------------------------------------------


def test_doramodel_trainable_parameters_exclude_weight0():
    """trainable_parameters() must not include any weight_0 buffers."""

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.proj = nn.Linear(8, 8)

        def forward(self, x):
            return self.proj(x)

    model = TinyModel()
    dora_model = DoRAModel(model, target_modules=["proj"], rank=2)

    trainable = dora_model.trainable_parameters()
    # All returned params must have requires_grad=True
    for p in trainable:
        assert p.requires_grad, "All trainable_parameters() entries must require grad"

    # weight_0 is a buffer and should not appear in parameters() at all
    param_data_ptrs = {p.data_ptr() for p in trainable}
    weight0 = dora_model.model.proj.weight_0
    assert weight0.data_ptr() not in param_data_ptrs, (
        "weight_0 must not appear in trainable_parameters()"
    )


# ---------------------------------------------------------------------------
# Test 12: Works with rank=1
# ---------------------------------------------------------------------------


def test_rank_one():
    """DoRALinear should work correctly with rank=1."""
    torch.manual_seed(7)
    linear = nn.Linear(12, 6, bias=True)
    dora = DoRALinear.from_linear(linear, rank=1)

    x = torch.randn(3, 12)
    with torch.no_grad():
        expected = linear(x)
        actual = dora(x)

    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
    assert dora.lora_A.shape == (1, 12)
    assert dora.lora_B.shape == (6, 1)


# ---------------------------------------------------------------------------
# Bonus Test 13: merge() returns correct type and shape attrs
# ---------------------------------------------------------------------------


def test_merge_shape_attributes(dora_from_linear):
    """Merged Linear must have in_features and out_features matching original."""
    merged = dora_from_linear.merge()
    assert merged.in_features == dora_from_linear.in_features
    assert merged.out_features == dora_from_linear.out_features
    assert merged.weight.shape == (dora_from_linear.out_features, dora_from_linear.in_features)


# ---------------------------------------------------------------------------
# Bonus Test 14: from_linear with no bias
# ---------------------------------------------------------------------------


def test_from_linear_no_bias():
    """from_linear with bias=False Linear must produce a DoRALinear without bias."""
    torch.manual_seed(5)
    linear = nn.Linear(10, 5, bias=False)
    dora = DoRALinear.from_linear(linear, rank=3)
    assert dora.bias is None, "DoRALinear should have no bias when source has none"

    x = torch.randn(4, 10)
    with torch.no_grad():
        expected = linear(x)
        actual = dora(x)
    torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
