"""Harmony response format parser and formatter.

Implements OpenAI's "harmony" structured conversation format used by gpt-oss-20b/120b.
Supports roles, tool calls, and structured outputs for training and inference.

Pure Python + PyTorch — no external APIs.
"""
from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable


# ── Data model ────────────────────────────────────────────────────────────────

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass
class ToolCall:
    """A single tool/function call requested by the model."""
    id: str
    type: str = "function"
    function_name: str = ""
    function_args: str = "{}"  # JSON string of arguments

    def __post_init__(self) -> None:
        if not self.id:
            self.id = str(uuid.uuid4())[:8]


@dataclass
class Message:
    """A single message in a harmony conversation."""
    role: MessageRole
    content: str | None = None        # None when tool_calls is populated
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_call_id: str | None = None   # for TOOL role responses
    name: str | None = None           # for TOOL role: function name


@dataclass
class Conversation:
    """A complete harmony conversation with optional tool schemas."""
    messages: list[Message]
    tools: list[dict] = field(default_factory=list)  # tool schemas (JSON)

    def system_message(self) -> str | None:
        """Return content of the first SYSTEM message, or None if absent."""
        for msg in self.messages:
            if msg.role == MessageRole.SYSTEM:
                return msg.content
        return None

    def user_messages(self) -> list[Message]:
        """Return all USER messages."""
        return [m for m in self.messages if m.role == MessageRole.USER]

    def assistant_messages(self) -> list[Message]:
        """Return all ASSISTANT messages."""
        return [m for m in self.messages if m.role == MessageRole.ASSISTANT]

    def last_assistant_message(self) -> Message | None:
        """Return the last ASSISTANT message, or None if absent."""
        for msg in reversed(self.messages):
            if msg.role == MessageRole.ASSISTANT:
                return msg
        return None


# ── Serialization ─────────────────────────────────────────────────────────────

def serialize_message(msg: Message) -> dict:
    """Convert a Message to a JSON-serializable dict."""
    data: dict = {"role": msg.role.value}

    if msg.content is not None:
        data["content"] = msg.content
    else:
        data["content"] = None

    if msg.tool_calls:
        data["tool_calls"] = [
            {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function_name,
                    "arguments": tc.function_args,
                },
            }
            for tc in msg.tool_calls
        ]
    else:
        data["tool_calls"] = []

    if msg.tool_call_id is not None:
        data["tool_call_id"] = msg.tool_call_id

    if msg.name is not None:
        data["name"] = msg.name

    return data


def deserialize_message(data: dict) -> Message:
    """Convert a dict back to a Message."""
    role = MessageRole(data["role"])
    content = data.get("content")

    tool_calls: list[ToolCall] = []
    for tc_data in data.get("tool_calls") or []:
        fn = tc_data.get("function", {})
        tool_calls.append(
            ToolCall(
                id=tc_data.get("id", ""),
                type=tc_data.get("type", "function"),
                function_name=fn.get("name", ""),
                function_args=fn.get("arguments", "{}"),
            )
        )

    return Message(
        role=role,
        content=content,
        tool_calls=tool_calls,
        tool_call_id=data.get("tool_call_id"),
        name=data.get("name"),
    )


def serialize_conversation(conv: Conversation) -> dict:
    """Convert a Conversation to a JSON-serializable dict."""
    return {
        "messages": [serialize_message(m) for m in conv.messages],
        "tools": list(conv.tools),
    }


def deserialize_conversation(data: dict) -> Conversation:
    """Convert a dict back to a Conversation."""
    messages = [deserialize_message(m) for m in data.get("messages", [])]
    tools = list(data.get("tools") or [])
    return Conversation(messages=messages, tools=tools)


# ── Formatting tokens ─────────────────────────────────────────────────────────

HARMONY_TOKENS: dict[str, str] = {
    "system_start": "<|system|>",
    "user_start": "<|user|>",
    "assistant_start": "<|assistant|>",
    "tool_start": "<|tool|>",
    "tool_call_start": "<|tool_call|>",
    "end_of_turn": "<|end_of_turn|>",
}

_ROLE_TO_TOKEN: dict[MessageRole, str] = {
    MessageRole.SYSTEM: HARMONY_TOKENS["system_start"],
    MessageRole.USER: HARMONY_TOKENS["user_start"],
    MessageRole.ASSISTANT: HARMONY_TOKENS["assistant_start"],
    MessageRole.TOOL: HARMONY_TOKENS["tool_start"],
}


def _format_message(msg: Message) -> str:
    """Format a single message as a harmony-formatted string."""
    role_token = _ROLE_TO_TOKEN[msg.role]
    eot = HARMONY_TOKENS["end_of_turn"]
    tc_token = HARMONY_TOKENS["tool_call_start"]

    if msg.tool_calls:
        # Assistant turn with tool calls: serialize each call as JSON inside tokens
        parts = [f"{role_token}\n"]
        for tc in msg.tool_calls:
            tc_dict = {
                "id": tc.id,
                "type": tc.type,
                "function": {
                    "name": tc.function_name,
                    "arguments": tc.function_args,
                },
            }
            parts.append(f"{tc_token}{json.dumps(tc_dict)}{eot}")
        return "".join(parts)
    else:
        content = msg.content or ""
        return f"{role_token}\n{content}{eot}"


def format_conversation_for_training(
    conv: Conversation,
    add_generation_prompt: bool = False,
) -> str:
    """Format a Conversation as a single string using harmony tokens.

    Each message is formatted as:
        {role_token}\\n{content}<|end_of_turn|>

    Tool calls are rendered as JSON within <|tool_call|> tags.

    Args:
        conv: The conversation to format.
        add_generation_prompt: If True, append "<|assistant|>\\n" at the end
            to prime generation.

    Returns:
        Formatted string ready for tokenization.
    """
    parts = [_format_message(m) for m in conv.messages]
    text = "".join(parts)
    if add_generation_prompt:
        text += HARMONY_TOKENS["assistant_start"] + "\n"
    return text


# ── SFT label building ────────────────────────────────────────────────────────

def build_sft_labels_harmony(
    formatted_text: str,
    tokenizer_encode: Callable[[str], list[int]],
    max_seq_len: int = 2048,
) -> tuple[list[int], list[int]]:
    """Build (input_ids, labels) for SFT training from a harmony-formatted string.

    Labels are -100 (masked) everywhere except assistant turns.  Assistant
    turns are identified as the text between "<|assistant|>\\n" and the
    following "<|end_of_turn|>".

    Args:
        formatted_text: Output of format_conversation_for_training().
        tokenizer_encode: Callable mapping str → list[int].
        max_seq_len: Maximum sequence length; truncated if longer.

    Returns:
        (input_ids, labels) as plain Python lists (same length).
    """
    assistant_start = HARMONY_TOKENS["assistant_start"] + "\n"
    eot = HARMONY_TOKENS["end_of_turn"]

    # Build a mask over the *characters* of formatted_text.
    # 1 = include in loss (assistant), 0 = mask to -100.
    char_mask = [0] * len(formatted_text)

    pos = 0
    while pos < len(formatted_text):
        idx = formatted_text.find(assistant_start, pos)
        if idx == -1:
            break
        # Content starts after the role token + newline
        content_start = idx + len(assistant_start)
        # Find the closing end_of_turn
        end_idx = formatted_text.find(eot, content_start)
        if end_idx == -1:
            end_idx = len(formatted_text)
        # Mark content + end_of_turn as "in loss"
        for i in range(content_start, end_idx + len(eot)):
            if i < len(char_mask):
                char_mask[i] = 1
        pos = end_idx + len(eot)

    # Now tokenize character by character to align tokens with the mask.
    # Since arbitrary tokenizers don't expose char→token alignment, we
    # tokenize each *segment* (consecutive same-mask chars) separately to
    # preserve the boundary correspondence.
    input_ids: list[int] = []
    labels: list[int] = []

    i = 0
    while i < len(formatted_text):
        # Find end of current same-mask segment
        current_mask = char_mask[i]
        j = i + 1
        while j < len(formatted_text) and char_mask[j] == current_mask:
            j += 1
        segment = formatted_text[i:j]
        seg_ids = tokenizer_encode(segment)
        input_ids.extend(seg_ids)
        if current_mask == 1:
            labels.extend(seg_ids)
        else:
            labels.extend([-100] * len(seg_ids))
        i = j

    # Truncate
    input_ids = input_ids[:max_seq_len]
    labels = labels[:max_seq_len]
    return input_ids, labels


# ── Validation ────────────────────────────────────────────────────────────────

def validate_conversation(conv: Conversation) -> list[str]:
    """Validate a Conversation and return a list of error strings.

    An empty list means the conversation is valid.

    Checks:
    - At least one message present.
    - First non-SYSTEM message must be USER or SYSTEM.
    - After stripping any leading SYSTEM message, messages must alternate
      USER / ASSISTANT (USER first).
    - TOOL messages must follow an ASSISTANT message that has tool_calls.
    """
    errors: list[str] = []
    msgs = conv.messages

    if not msgs:
        errors.append("Conversation has no messages.")
        return errors

    # Strip leading SYSTEM messages
    non_system = [m for m in msgs if m.role != MessageRole.SYSTEM]

    # The very first message must be SYSTEM or USER
    if msgs[0].role not in (MessageRole.SYSTEM, MessageRole.USER):
        errors.append(
            f"First message must be SYSTEM or USER, got {msgs[0].role.value!r}."
        )

    if not non_system:
        # Only system messages — arguably fine but warn
        errors.append("Conversation contains only SYSTEM messages.")
        return errors

    if non_system[0].role != MessageRole.USER:
        errors.append(
            f"First non-system message must be USER, got {non_system[0].role.value!r}."
        )

    # Check alternation among non-TOOL messages (TOOL messages interleave
    # after ASSISTANT tool-call turns and are allowed there).
    exchange: list[Message] = [m for m in non_system if m.role != MessageRole.TOOL]
    for i, msg in enumerate(exchange):
        expected_role = MessageRole.USER if i % 2 == 0 else MessageRole.ASSISTANT
        if msg.role != expected_role:
            errors.append(
                f"Message {i} in exchange: expected {expected_role.value!r}, "
                f"got {msg.role.value!r}."
            )

    # TOOL messages must be preceded by an ASSISTANT with tool_calls
    for i, msg in enumerate(msgs):
        if msg.role == MessageRole.TOOL:
            if i == 0:
                errors.append("TOOL message at position 0 has no preceding ASSISTANT.")
            else:
                prev = msgs[i - 1]
                if prev.role != MessageRole.ASSISTANT or not prev.tool_calls:
                    errors.append(
                        f"TOOL message at position {i} must follow an ASSISTANT "
                        f"message with tool_calls."
                    )

    return errors


# ── Tool call extraction ──────────────────────────────────────────────────────

_TOOL_CALL_RE = re.compile(
    r"<\|tool_call\|>(.*?)<\|end_of_turn\|>",
    re.DOTALL,
)


def extract_tool_calls_from_text(text: str) -> list[ToolCall]:
    """Parse tool calls from generated text containing <|tool_call|> tags.

    Each <|tool_call|>...<|end_of_turn|> block is expected to contain a
    JSON object with the same schema as the OpenAI tool-call format:
        {"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}

    Args:
        text: Raw generated text potentially containing tool call blocks.

    Returns:
        List of ToolCall objects parsed from the text.
    """
    calls: list[ToolCall] = []
    for match in _TOOL_CALL_RE.finditer(text):
        raw_json = match.group(1).strip()
        try:
            data = json.loads(raw_json)
        except json.JSONDecodeError:
            continue
        fn = data.get("function", {})
        calls.append(
            ToolCall(
                id=data.get("id", str(uuid.uuid4())[:8]),
                type=data.get("type", "function"),
                function_name=fn.get("name", ""),
                function_args=fn.get("arguments", "{}"),
            )
        )
    return calls


# ── Harmony response parsing ──────────────────────────────────────────────────

_ROLE_TOKEN_RE = re.compile(
    r"<\|(?:system|user|assistant|tool)\|>",
)

_ROLE_NAMES = {
    HARMONY_TOKENS["system_start"]: "system",
    HARMONY_TOKENS["user_start"]: "user",
    HARMONY_TOKENS["assistant_start"]: "assistant",
    HARMONY_TOKENS["tool_start"]: "tool",
}


def parse_harmony_response(text: str) -> dict:
    """Parse a complete harmony-formatted response string.

    Expects text of the form:
        <|{role}|>\\n{content}<|end_of_turn|>

    If tool calls are present inside the content, they are extracted.

    Args:
        text: Raw harmony-formatted string (single message or fragment).

    Returns:
        {"role": str, "content": str | None, "tool_calls": list[dict]}
    """
    eot = HARMONY_TOKENS["end_of_turn"]
    role: str = "assistant"
    content: str | None = None
    tool_calls_raw: list[ToolCall] = []

    # Identify role token
    for token, role_name in _ROLE_NAMES.items():
        if text.startswith(token):
            role = role_name
            # Strip role token (and optional leading newline)
            remainder = text[len(token):]
            if remainder.startswith("\n"):
                remainder = remainder[1:]
            # Strip trailing end_of_turn
            if eot in remainder:
                remainder = remainder[: remainder.rfind(eot)]
            remainder = remainder.rstrip()

            # Check for embedded tool calls
            tool_calls_raw = extract_tool_calls_from_text(
                text  # search original text for full tags
            )
            if tool_calls_raw:
                content = None
            else:
                content = remainder if remainder else None
            break
    else:
        # No recognized role token — treat whole text as assistant content
        cleaned = text
        if eot in cleaned:
            cleaned = cleaned[: cleaned.rfind(eot)]
        content = cleaned.strip() or None

    tool_calls_list = [
        {
            "id": tc.id,
            "type": tc.type,
            "function": {"name": tc.function_name, "arguments": tc.function_args},
        }
        for tc in tool_calls_raw
    ]

    return {"role": role, "content": content, "tool_calls": tool_calls_list}
