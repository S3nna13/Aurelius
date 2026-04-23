"""Unit tests for ``src/model/head_registry.py``."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.model.head_registry import (
    HEAD_REGISTRY,
    HeadFactoryError,
    HeadKind,
    HeadSpec,
    build_head,
    get_head,
    heads_by_kind,
    list_heads,
    register_head,
)


def _make_spec(**overrides) -> HeadSpec:
    kwargs = dict(
        id=f"test/spec-{len(HEAD_REGISTRY)}",
        kind=HeadKind.LM,
        output_dim=16,
        bias=False,
        tied_to_embedding=False,
    )
    kwargs.update(overrides)
    return HeadSpec(**kwargs)


def test_all_head_kinds_defined() -> None:
    expected = {
        "LM",
        "REWARD",
        "CLASSIFIER",
        "DUAL",
        "EMBEDDING",
        "VALUE",
        "MULTI_HEAD",
    }
    assert {k.name for k in HeadKind} == expected


def test_register_and_lookup() -> None:
    spec = _make_spec(id="test/register-and-lookup", output_dim=32)
    register_head(spec)
    assert get_head("test/register-and-lookup") is spec
    assert "test/register-and-lookup" in list_heads()


def test_duplicate_registration_raises() -> None:
    spec = _make_spec(id="test/dup")
    register_head(spec)
    with pytest.raises(HeadFactoryError):
        register_head(spec)


def test_get_missing_raises() -> None:
    with pytest.raises(HeadFactoryError):
        get_head("does/not/exist")


def test_heads_by_kind_filters() -> None:
    lm_heads = heads_by_kind(HeadKind.LM)
    assert all(s.kind is HeadKind.LM for s in lm_heads)
    reward_heads = heads_by_kind(HeadKind.REWARD)
    assert all(s.kind is HeadKind.REWARD for s in reward_heads)


def test_build_head_lm_returns_linear() -> None:
    spec = _make_spec(id="test/build-lm", output_dim=64)
    head = build_head(spec, d_model=8)
    assert isinstance(head, nn.Linear)
    assert head.out_features == 64
    assert head.in_features == 8


def test_build_head_dual_has_both_outputs() -> None:
    spec = HeadSpec(
        id="test/dual",
        kind=HeadKind.DUAL,
        output_dim=50,
        bias=True,
    )
    head = build_head(spec, d_model=12)
    assert hasattr(head, "lm_logits")
    assert hasattr(head, "value")
    assert isinstance(head.lm_logits, nn.Linear)
    assert isinstance(head.value, nn.Linear)
    assert head.value.out_features == 1


def test_build_head_multi_head_has_module_dict() -> None:
    spec = HeadSpec(
        id="test/multi",
        kind=HeadKind.MULTI_HEAD,
        output_dim=4,
        metadata={"subhead_names": ["task_a", "task_b"]},
    )
    head = build_head(spec, d_model=8)
    assert isinstance(head.heads, nn.ModuleDict)
    assert "task_a" in head.heads
    assert "task_b" in head.heads


def test_reward_output_dim_must_be_one() -> None:
    with pytest.raises(HeadFactoryError):
        HeadSpec(id="test/bad-reward", kind=HeadKind.REWARD, output_dim=2)


def test_classifier_output_dim_positive() -> None:
    with pytest.raises(HeadFactoryError):
        HeadSpec(id="test/bad-classifier", kind=HeadKind.CLASSIFIER, output_dim=0)
    with pytest.raises(HeadFactoryError):
        HeadSpec(id="test/bad-classifier2", kind=HeadKind.CLASSIFIER, output_dim=-1)


def test_tied_to_embedding_propagated_in_metadata() -> None:
    spec = HeadSpec(
        id="test/tied-lm",
        kind=HeadKind.LM,
        output_dim=1024,
        tied_to_embedding=True,
    )
    assert spec.tied_to_embedding is True
    assert spec.metadata.get("tied_to_embedding") is True


def test_seed_heads_present() -> None:
    for seed_id in (
        "aurelius/default-lm",
        "aurelius/reward-v1",
        "aurelius/classifier-binary",
        "aurelius/value-head",
    ):
        assert seed_id in HEAD_REGISTRY


def test_unicode_id_allowed() -> None:
    spec = HeadSpec(id="tēst/ユニコード", kind=HeadKind.LM, output_dim=4)
    register_head(spec)
    assert get_head("tēst/ユニコード") is spec


def test_seed_determinism() -> None:
    lm = get_head("aurelius/default-lm")
    assert lm.output_dim == 128000
    assert lm.tied_to_embedding is True
    reward = get_head("aurelius/reward-v1")
    assert reward.output_dim == 1
    assert reward.kind is HeadKind.REWARD


def test_build_head_forward_shape() -> None:
    spec = HeadSpec(id="test/forward-shape", kind=HeadKind.LM, output_dim=17)
    d_model = 9
    head = build_head(spec, d_model=d_model)
    batch, seq = 2, 3
    x = torch.randn(batch, seq, d_model)
    y = head(x)
    assert y.shape == (batch, seq, 17)


def test_multi_head_requires_subhead_names() -> None:
    with pytest.raises(HeadFactoryError):
        HeadSpec(id="test/multi-bad", kind=HeadKind.MULTI_HEAD, output_dim=4)


def test_value_output_dim_must_be_one() -> None:
    with pytest.raises(HeadFactoryError):
        HeadSpec(id="test/bad-value", kind=HeadKind.VALUE, output_dim=3)


def test_tied_to_embedding_only_for_lm() -> None:
    with pytest.raises(HeadFactoryError):
        HeadSpec(
            id="test/bad-tied",
            kind=HeadKind.REWARD,
            output_dim=1,
            tied_to_embedding=True,
        )
