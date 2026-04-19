"""Unit tests for the Harmony chat template."""

from __future__ import annotations

import pytest

from src.chat.harmony_template import (
    HarmonyFormatError,
    HarmonyMessage,
    HarmonyTemplate,
)


@pytest.fixture
def tpl() -> HarmonyTemplate:
    return HarmonyTemplate()


def test_encode_single_user(tpl: HarmonyTemplate) -> None:
    out = tpl.encode([HarmonyMessage(role="user", content="hello")])
    assert out == "<|start|>user<|message|>hello<|end|>"


def test_encode_assistant_analysis(tpl: HarmonyTemplate) -> None:
    out = tpl.encode(
        [HarmonyMessage(role="assistant", content="think", channel="analysis")]
    )
    assert (
        out
        == "<|start|>assistant<|channel|>analysis<|message|>think<|end|>"
    )


def test_encode_assistant_final(tpl: HarmonyTemplate) -> None:
    out = tpl.encode(
        [HarmonyMessage(role="assistant", content="answer", channel="final")]
    )
    assert (
        out
        == "<|start|>assistant<|channel|>final<|message|>answer<|end|>"
    )


def test_encode_multi_turn(tpl: HarmonyTemplate) -> None:
    msgs = [
        HarmonyMessage(role="system", content="sys"),
        HarmonyMessage(role="developer", content="dev"),
        HarmonyMessage(role="user", content="q"),
        HarmonyMessage(
            role="assistant", content="reason", channel="analysis"
        ),
        HarmonyMessage(role="assistant", content="ans", channel="final"),
    ]
    out = tpl.encode(msgs)
    assert out == (
        "<|start|>system<|message|>sys<|end|>"
        "<|start|>developer<|message|>dev<|end|>"
        "<|start|>user<|message|>q<|end|>"
        "<|start|>assistant<|channel|>analysis<|message|>reason<|end|>"
        "<|start|>assistant<|channel|>final<|message|>ans<|end|>"
    )


def test_round_trip_preserves_roles_channels_content(
    tpl: HarmonyTemplate,
) -> None:
    msgs = [
        HarmonyMessage(role="system", content="you are helpful"),
        HarmonyMessage(role="developer", content="tools: none"),
        HarmonyMessage(role="user", content="multi\nline\nquestion?"),
        HarmonyMessage(
            role="assistant",
            content="let me think about it",
            channel="analysis",
        ),
        HarmonyMessage(
            role="assistant",
            content="here is my answer",
            channel="final",
        ),
        HarmonyMessage(role="tool", content="{\"result\": 42}"),
    ]
    text = tpl.encode(msgs)
    decoded = tpl.decode(text)
    assert len(decoded) == len(msgs)
    for original, got in zip(msgs, decoded):
        assert got.role == original.role
        assert got.content == original.content
        assert got.channel == original.channel


def test_invalid_role_raises(tpl: HarmonyTemplate) -> None:
    with pytest.raises(HarmonyFormatError):
        tpl.encode([HarmonyMessage(role="root", content="x")])


def test_invalid_channel_for_assistant_raises(tpl: HarmonyTemplate) -> None:
    with pytest.raises(HarmonyFormatError):
        tpl.encode(
            [
                HarmonyMessage(
                    role="assistant", content="x", channel="bogus"
                )
            ]
        )


def test_channel_on_non_assistant_raises(tpl: HarmonyTemplate) -> None:
    with pytest.raises(HarmonyFormatError):
        tpl.encode(
            [HarmonyMessage(role="user", content="x", channel="final")]
        )


@pytest.mark.parametrize(
    "token",
    ["<|start|>", "<|end|>", "<|message|>", "<|channel|>"],
)
def test_control_token_injection_raises(
    tpl: HarmonyTemplate, token: str
) -> None:
    with pytest.raises(HarmonyFormatError):
        tpl.encode(
            [HarmonyMessage(role="user", content=f"prefix{token}suffix")]
        )


def test_add_generation_prompt_appends_open_assistant(
    tpl: HarmonyTemplate,
) -> None:
    out = tpl.encode(
        [HarmonyMessage(role="user", content="hi")],
        add_generation_prompt=True,
    )
    assert out.endswith("<|start|>assistant")
    # No closing tag / channel / message marker after the open prompt.
    tail = out[out.rindex("<|start|>assistant"):]
    assert "<|end|>" not in tail
    assert "<|message|>" not in tail
    assert "<|channel|>" not in tail


def test_determinism(tpl: HarmonyTemplate) -> None:
    msgs = [
        HarmonyMessage(role="system", content="s"),
        HarmonyMessage(role="user", content="u"),
        HarmonyMessage(role="assistant", content="a", channel="final"),
    ]
    a = tpl.encode(msgs)
    b = tpl.encode(msgs)
    c = tpl.encode(msgs)
    assert a == b == c


def test_channel_none_defaults_to_final_for_assistant(
    tpl: HarmonyTemplate,
) -> None:
    out = tpl.encode(
        [HarmonyMessage(role="assistant", content="hi", channel=None)]
    )
    assert "<|channel|>final<|message|>hi" in out


def test_empty_messages(tpl: HarmonyTemplate) -> None:
    assert tpl.encode([]) == ""
    assert tpl.decode("") == []
    # Generation-prompt-only is permitted on an empty transcript.
    gp = tpl.encode([], add_generation_prompt=True)
    assert gp == "<|start|>assistant"
    assert tpl.decode(gp) == []


def test_siblings_still_import_without_regression() -> None:
    # Guard against accidental edits that would break peer templates.
    from src.chat.chatml_template import ChatMLTemplate, Message
    from src.chat.llama3_template import Llama3Template

    chatml = ChatMLTemplate()
    llama = Llama3Template()
    msgs = [
        Message(role="user", content="hi"),
        Message(role="assistant", content="hello"),
    ]
    chatml_out = chatml.encode(msgs)
    assert "<|im_start|>user" in chatml_out
    assert "<|im_end|>" in chatml_out
    assert chatml.decode(chatml_out) == msgs

    llama_out = llama.encode(msgs)
    assert "<|begin_of_text|>" in llama_out
    assert "<|start_header_id|>user<|end_header_id|>" in llama_out
    assert llama.decode(llama_out) == msgs


def test_encode_token_ids(tpl: HarmonyTemplate) -> None:
    msgs = [HarmonyMessage(role="user", content="hi")]
    ids = tpl.encode_token_ids(msgs, tokenizer=lambda s: [ord(c) for c in s])
    assert ids and all(isinstance(i, int) for i in ids)


def test_decode_with_trailing_open_generation_prompt(
    tpl: HarmonyTemplate,
) -> None:
    text = (
        "<|start|>user<|message|>hi<|end|>"
        "<|start|>assistant"
    )
    got = tpl.decode(text)
    assert len(got) == 1
    assert got[0].role == "user" and got[0].content == "hi"
