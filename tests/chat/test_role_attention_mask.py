"""Unit tests for src/chat/role_attention_mask.py."""

from __future__ import annotations

import pytest
import torch

from src.chat.role_attention_mask import (
    MASK_VALUE,
    RoleSpan,
    RoleSpanError,
    build_loss_mask,
    build_role_mask,
    validate_spans,
)

S = 16


def _full_spans() -> list[RoleSpan]:
    # system[0:3] user[3:6] assistant[6:9] tool[9:12] user[12:14] assistant[14:16]
    return [
        RoleSpan("system", 0, 3),
        RoleSpan("user", 3, 6),
        RoleSpan("assistant", 6, 9),
        RoleSpan("tool", 9, 12),
        RoleSpan("user", 12, 14),
        RoleSpan("assistant", 14, 16),
    ]


def test_causal_shape_and_upper_triangle_masked():
    spans = [RoleSpan("user", 0, S)]
    m = build_role_mask(spans, S, system_priority=False, causal=True)
    assert m.shape == (S, S)
    for i in range(S):
        for j in range(S):
            if j > i:
                assert m[i, j].item() == pytest.approx(MASK_VALUE)
            else:
                assert m[i, j].item() == 0.0


def test_system_priority_all_rows_see_system():
    spans = _full_spans()
    m = build_role_mask(spans, S, system_priority=True, causal=True)
    # System columns 0..3 must be attendable by every row.
    for i in range(S):
        for j in range(0, 3):
            assert m[i, j].item() == 0.0


def test_validate_spans_rejects_overlap():
    spans = [RoleSpan("system", 0, 5), RoleSpan("user", 4, 10), RoleSpan("assistant", 10, S)]
    with pytest.raises(RoleSpanError):
        validate_spans(spans, S)


def test_validate_spans_rejects_gap():
    spans = [RoleSpan("system", 0, 3), RoleSpan("user", 5, 10), RoleSpan("assistant", 10, S)]
    with pytest.raises(RoleSpanError):
        validate_spans(spans, S)


def test_validate_spans_rejects_out_of_range():
    spans = [RoleSpan("system", 0, 3), RoleSpan("user", 3, S + 1)]
    with pytest.raises(RoleSpanError):
        validate_spans(spans, S)


def test_validate_spans_rejects_unknown_role():
    spans = [RoleSpan("robot", 0, S)]
    with pytest.raises(RoleSpanError):
        validate_spans(spans, S)


def test_build_loss_mask_only_assistant_positions():
    spans = _full_spans()
    loss = build_loss_mask(spans, S, loss_roles=("assistant",))
    expected = torch.zeros(S, dtype=torch.bool)
    expected[6:9] = True
    expected[14:16] = True
    assert torch.equal(loss, expected)


def test_loss_mask_length_and_dtype():
    spans = _full_spans()
    loss = build_loss_mask(spans, S)
    assert loss.shape == (S,)
    assert loss.dtype == torch.bool


def test_multi_role_sequence_mask_well_formed():
    spans = _full_spans()
    m = build_role_mask(spans, S, system_priority=True, causal=True)
    # Causal lower triangle (excluding system cols) below diagonal: 0.0.
    for i in range(S):
        # A row should be able to attend to itself (diagonal).
        assert m[i, i].item() == 0.0
    # Upper triangle outside system cols must be masked.
    for i in range(S):
        for j in range(i + 1, S):
            if j < 3:  # system col exempt
                continue
            assert m[i, j].item() == pytest.approx(MASK_VALUE)


def test_user_tool_barrier():
    spans = _full_spans()
    build_role_mask(spans, S, system_priority=True, causal=True)
    # user[12:14] must not attend tool[9:12] (tool is before, so actually
    # *only* later tools are blocked -- here tool 9:12 came before user
    # 12:14, so allowed). Ensure user[3:6] cannot attend tool[9:12] (but
    # causal already masks that). Create a case where tool is AFTER user
    # but inside sequence: the first user 3:6 has later tool 9:12.
    # Causal already masks those. The barrier is redundant with causal
    # when causal=True, but must hold under non-causal too.
    m2 = build_role_mask(spans, S, system_priority=True, causal=False)
    for i in range(3, 6):  # user[3:6]
        for j in range(9, 12):  # tool[9:12]
            assert m2[i, j].item() == pytest.approx(MASK_VALUE)


def test_non_causal_mode_respects_system_priority():
    spans = _full_spans()
    m = build_role_mask(spans, S, system_priority=True, causal=False)
    # All system cols attendable everywhere.
    for i in range(S):
        for j in range(0, 3):
            assert m[i, j].item() == 0.0
    # Non-causal: assistant row 6 can attend future user 12 (no barrier
    # applies here).
    assert m[6, 12].item() == 0.0


def test_dtype_and_device_honored():
    spans = _full_spans()
    m = build_role_mask(spans, S, dtype=torch.float16, device=torch.device("cpu"))
    assert m.dtype == torch.float16
    assert m.device.type == "cpu"


def test_empty_spans_raises():
    with pytest.raises(RoleSpanError):
        build_role_mask([], S)
    with pytest.raises(RoleSpanError):
        build_loss_mask([], S)


def test_determinism():
    spans = _full_spans()
    a = build_role_mask(spans, S)
    b = build_role_mask(spans, S)
    assert torch.equal(a, b)
    la = build_loss_mask(spans, S)
    lb = build_loss_mask(spans, S)
    assert torch.equal(la, lb)


def test_loss_mask_custom_roles():
    spans = _full_spans()
    loss = build_loss_mask(spans, S, loss_roles=("assistant", "tool"))
    expected = torch.zeros(S, dtype=torch.bool)
    expected[6:9] = True
    expected[9:12] = True
    expected[14:16] = True
    assert torch.equal(loss, expected)
