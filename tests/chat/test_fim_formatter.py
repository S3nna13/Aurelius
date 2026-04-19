"""Unit tests for src.chat.fim_formatter."""

from __future__ import annotations

import random

import pytest

from src.chat.fim_formatter import (
    FIM_MIDDLE,
    FIM_PAD,
    FIM_PREFIX,
    FIM_SUFFIX,
    FIMExample,
    FIMFormatError,
    FIMFormatter,
)


def test_psm_format_contains_tokens_in_order() -> None:
    f = FIMFormatter(mode="psm")
    out = f.format(FIMExample(prefix="A", middle="B", suffix="C"))
    assert out == f"{FIM_PREFIX}A{FIM_SUFFIX}C{FIM_MIDDLE}B"
    # order check
    i_p = out.index(FIM_PREFIX)
    i_s = out.index(FIM_SUFFIX)
    i_m = out.index(FIM_MIDDLE)
    assert i_p < i_s < i_m


def test_spm_format_differs_from_psm() -> None:
    ex = FIMExample(prefix="A", middle="B", suffix="C")
    psm = FIMFormatter(mode="psm").format(ex)
    spm = FIMFormatter(mode="spm").format(ex)
    assert psm != spm
    assert spm == f"{FIM_SUFFIX}C{FIM_PREFIX}A{FIM_MIDDLE}B"


@pytest.mark.parametrize("mode", ["psm", "spm"])
def test_roundtrip_format_parse(mode: str) -> None:
    f = FIMFormatter(mode=mode)
    ex = FIMExample(prefix="def foo():\n    ", middle="return 1", suffix="\n")
    text = f.format(ex)
    parsed = f.parse(text)
    assert parsed == ex


def test_format_for_inference_omits_middle() -> None:
    f = FIMFormatter(mode="psm")
    out = f.format_for_inference(prefix="pre", suffix="suf")
    assert out.endswith(FIM_MIDDLE)
    assert out == f"{FIM_PREFIX}pre{FIM_SUFFIX}suf{FIM_MIDDLE}"


def test_unknown_mode_raises() -> None:
    with pytest.raises(FIMFormatError):
        FIMFormatter(mode="xyz")


@pytest.mark.parametrize("field", ["prefix", "middle", "suffix"])
@pytest.mark.parametrize("tok", [FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE, FIM_PAD])
def test_control_token_injection_raises(field: str, tok: str) -> None:
    kwargs = {"prefix": "p", "middle": "m", "suffix": "s"}
    kwargs[field] = f"x{tok}y"
    f = FIMFormatter(mode="psm")
    with pytest.raises(FIMFormatError):
        f.format(FIMExample(**kwargs))


def test_include_middle_false_matches_inference() -> None:
    f_train = FIMFormatter(mode="psm", include_middle=False)
    f_infer = FIMFormatter(mode="psm")
    ex = FIMExample(prefix="p", middle="will-be-dropped", suffix="s")
    assert f_train.format(ex) == f_infer.format_for_inference("p", "s")


def test_loss_mask_length_equals_tokens() -> None:
    toks = [1, 2, 3, 99, 4, 5]
    mask = FIMFormatter.make_loss_mask(toks, middle_token_id=99)
    assert len(mask) == len(toks)


def test_loss_mask_true_only_after_middle() -> None:
    toks = [10, 11, 99, 20, 21]
    mask = FIMFormatter.make_loss_mask(toks, middle_token_id=99)
    assert mask == [False, False, False, True, True]


def test_loss_mask_all_false_when_middle_absent() -> None:
    toks = [1, 2, 3]
    mask = FIMFormatter.make_loss_mask(toks, middle_token_id=99)
    assert mask == [False, False, False]


def test_empty_spans_work() -> None:
    f = FIMFormatter(mode="psm")
    ex = FIMExample(prefix="", middle="", suffix="")
    text = f.format(ex)
    assert text == f"{FIM_PREFIX}{FIM_SUFFIX}{FIM_MIDDLE}"
    assert f.parse(text) == ex


def test_determinism() -> None:
    f = FIMFormatter(mode="psm")
    ex = FIMExample(prefix="a", middle="b", suffix="c")
    a = f.format(ex)
    b = f.format(ex)
    assert a == b


def test_random_mode_produces_both_orderings() -> None:
    rng = random.Random(0xA11CE)
    f = FIMFormatter(random_mode=True, rng=rng)
    ex = FIMExample(prefix="p", middle="m", suffix="s")
    outs = [f.format(ex) for _ in range(20)]
    saw_psm = any(o.startswith(FIM_PREFIX) for o in outs)
    saw_spm = any(o.startswith(FIM_SUFFIX) for o in outs)
    assert saw_psm and saw_spm


def test_parse_malformed_raises() -> None:
    f = FIMFormatter(mode="psm")
    with pytest.raises(FIMFormatError):
        f.parse("no sentinels here")
    with pytest.raises(FIMFormatError):
        f.parse(f"{FIM_PREFIX}only-prefix")
    with pytest.raises(FIMFormatError):
        f.parse(f"{FIM_PREFIX}p{FIM_SUFFIX}s")  # missing middle


def test_parse_rejects_non_str() -> None:
    f = FIMFormatter(mode="psm")
    with pytest.raises(FIMFormatError):
        f.parse(123)  # type: ignore[arg-type]


def test_format_rejects_non_example() -> None:
    f = FIMFormatter(mode="psm")
    with pytest.raises(FIMFormatError):
        f.format("not an example")  # type: ignore[arg-type]


def test_loss_mask_rejects_non_list() -> None:
    with pytest.raises(FIMFormatError):
        FIMFormatter.make_loss_mask((1, 2, 3), middle_token_id=2)  # type: ignore[arg-type]
