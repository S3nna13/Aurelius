"""Integration tests for src.chat.instruction_tuning_data.

Verifies the generator surface is exposed through src.chat and that a
generated sample round-trips cleanly through the ChatML template
encoder.
"""

from __future__ import annotations


def test_generators_exposed_from_src_chat():
    import src.chat as chat

    assert hasattr(chat, "InstructionSample")
    assert hasattr(chat, "MagpieGenerator")
    assert hasattr(chat, "SelfInstructGenerator")
    assert hasattr(chat, "EvolInstructGenerator")


def test_generated_sample_roundtrips_through_chatml():
    from src.chat import (
        ChatMLTemplate,
        Message,
        MagpieGenerator,
    )

    counter = {"n": 0}

    def fake_lm(prompt: str) -> str:
        counter["n"] += 1
        return f"payload-{counter['n']}"

    gen = MagpieGenerator(fake_lm)
    samples = gen.generate(n=1, seed=7)
    assert len(samples) == 1
    s = samples[0]

    tpl = ChatMLTemplate()
    # Should not raise.
    wire = tpl.encode(
        [
            Message(role="user", content=s.instruction),
            Message(role="assistant", content=s.response),
        ]
    )
    decoded = tpl.decode(wire)
    assert len(decoded) == 2
    assert decoded[0].content == s.instruction
    assert decoded[1].content == s.response
