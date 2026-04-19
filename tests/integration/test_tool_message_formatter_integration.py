"""End-to-end integration: tool-result formatter + ChatML template.

Verifies that a ``ToolMessageFormatter`` pulled from
``MESSAGE_FORMAT_REGISTRY`` produces messages that round-trip through
the ``ChatMLTemplate`` in ``CHAT_TEMPLATE_REGISTRY`` without raising.
"""

from __future__ import annotations

from src.chat import CHAT_TEMPLATE_REGISTRY, MESSAGE_FORMAT_REGISTRY
from src.chat.chatml_template import ChatMLTemplate, Message
from src.chat.tool_message_formatter import ToolMessageFormatter, ToolResult


def test_registry_wiring_chatml_roundtrip():
    chatml = CHAT_TEMPLATE_REGISTRY["chatml"]
    assert isinstance(chatml, ChatMLTemplate)

    FormatterCls = MESSAGE_FORMAT_REGISTRY["tool_result"]
    assert FormatterCls is ToolMessageFormatter

    fmt = FormatterCls(template="chatml", max_content_chars=256)
    results = [
        ToolResult(call_id="call_1", name="read_file", content="file contents\nline2"),
        ToolResult(
            call_id="call_2",
            name="bash",
            content="command failed: permission denied",
            is_error=True,
        ),
    ]

    messages = fmt.format_batch(results)
    assert all(isinstance(m, Message) for m in messages)
    assert [m.role for m in messages] == ["tool", "tool"]

    # Encode via the registry-provided template instance.
    wire = chatml.encode(messages)
    assert "[tool: read_file id=call_1]" in wire
    assert "[ERROR] command failed" in wire

    # Round-trip: decode must recover the same roles and preserve the
    # call_id markers inside content.
    decoded = chatml.decode(wire)
    assert [d.role for d in decoded] == ["tool", "tool"]
    assert "id=call_1" in decoded[0].content
    assert "id=call_2" in decoded[1].content
    assert "[ERROR]" in decoded[1].content


def test_to_prompt_turn_uses_registry_template():
    chatml = CHAT_TEMPLATE_REGISTRY["chatml"]
    fmt = MESSAGE_FORMAT_REGISTRY["tool_result"](template="chatml")
    rendered = fmt.to_prompt_turn(
        [ToolResult(call_id="c", name="t", content="ok")],
        template_obj=chatml,
    )
    # Must be a non-empty, well-formed ChatML turn.
    assert rendered.startswith("<|im_start|>tool\n")
    assert rendered.endswith("<|im_end|>\n")
    # And parsing it does not raise.
    parsed = chatml.decode(rendered)
    assert len(parsed) == 1
    assert parsed[0].role == "tool"
