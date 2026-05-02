"""Compatibility tokenizer surface used by the comprehensive test harness.

This module intentionally keeps the implementation small and deterministic:
- byte-level reversible encoding for arbitrary text
- named special tokens with stable ids
- simple reasoning / tool-call / memory / agent helpers

It does not replace the production ``src.data.tokenizer`` implementation.
"""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "TokenizerConfig",
    "AureliusTokenizer",
    "ReasoningTokenizer",
    "ToolCallTokenizer",
    "MemoryTokenizer",
    "AgentRoutingTokenizer",
]


@dataclass(slots=True)
class TokenizerConfig:
    """Minimal configuration for the compatibility tokenizer."""

    vocab_size: int = 128_000
    model_max_length: int = 8192


_SPECIAL_TOKEN_IDS: dict[str, int] = {
    "<|think|>": 256,
    "</|think|>": 257,
    "<|deep_think|>": 258,
    "</|deep_think|>": 259,
    "<|plan|>": 260,
    "</|plan|>": 261,
    "<|scratchpad|>": 262,
    "</|scratchpad|>": 263,
    "<|final_answer|>": 264,
    "</|final_answer|>": 265,
    "<|verify|>": 266,
    "</|verify|>": 267,
    "<|self_check|>": 268,
    "</|self_check|>": 269,
    "<|conclusion|>": 270,
    "</|conclusion|>": 271,
    "<|memory_store|>": 280,
    "<|memory_retrieve|>": 281,
    "<|reference|>": 282,
    "<|agent|>": 290,
    "<|delegate|>": 291,
    "<|agent_result|>": 292,
    "<|swarm|>": 293,
    "<|debate|>": 294,
    "<|pad|>": 295,
    "<|bos|>": 296,
    "<|eos|>": 297,
    "<|unk|>": 298,
    "<|system|>": 299,
    "<|tool_calls|>": 300,
    "<|invoke|>": 301,
    "<|parameter|>": 302,
    "<|user|>": 303,
    "<|assistant|>": 304,
    "<|end|>": 305,
    "<|tool_call|>": 306,
    "<|tool_result|>": 307,
}

_SPECIAL_TOKENS_SORTED = sorted(_SPECIAL_TOKEN_IDS, key=len, reverse=True)
_INVOKE_RE = re.compile(r'<\|invoke\|>\s*name="([^"]+)">(.*?)</\|invoke\|>', re.DOTALL)
_PARAM_RE = re.compile(
    r'<\|parameter\|>\s*name="([^"]+)"\s*string="(true|false)">(.*?)</\|parameter\|>',
    re.DOTALL,
)


class AureliusTokenizer:
    """Tiny reversible tokenizer with stable named special-token ids."""

    def __init__(self, cfg: TokenizerConfig | None = None) -> None:
        self.cfg = cfg or TokenizerConfig()
        self.vocab: dict[str, int] = dict(_SPECIAL_TOKEN_IDS)
        self.special_tokens: dict[str, int] = dict(_SPECIAL_TOKEN_IDS)
        self._id_to_token: dict[int, str] = {v: k for k, v in self.vocab.items()}

    @property
    def vocab_size(self) -> int:
        return max(self.cfg.vocab_size, max(self.vocab.values(), default=255) + 1)

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            matched = False
            for token in _SPECIAL_TOKENS_SORTED:
                if text.startswith(token, i):
                    ids.append(self.vocab[token])
                    i += len(token)
                    matched = True
                    break
            if matched:
                continue
            for byte in text[i].encode("utf-8"):
                ids.append(byte)
            i += 1
        return ids

    def decode(self, ids: list[int]) -> str:
        chunks: list[str] = []
        buffer = bytearray()
        for token_id in ids:
            token = self._id_to_token.get(token_id)
            if token is not None:
                if buffer:
                    chunks.append(buffer.decode("utf-8", errors="replace"))
                    buffer.clear()
                chunks.append(token)
                continue
            if 0 <= token_id <= 255:
                buffer.append(token_id)
        if buffer:
            chunks.append(buffer.decode("utf-8", errors="replace"))
        return "".join(chunks)

    def count_tokens(self, text: str) -> int:
        return len(self.encode(text))

    def compression_ratio(self, text: str) -> float:
        token_count = max(1, self.count_tokens(text))
        return len(text.encode("utf-8")) / token_count

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "cfg": asdict(self.cfg),
            "vocab": self.vocab,
            "special_tokens": self.special_tokens,
        }
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> AureliusTokenizer:
        path = Path(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        cfg = TokenizerConfig(**payload.get("cfg", {}))
        obj = cls(cfg)
        obj.vocab = {str(k): int(v) for k, v in payload.get("vocab", {}).items()}
        obj.special_tokens = {str(k): int(v) for k, v in payload.get("special_tokens", {}).items()}
        obj._id_to_token = {v: k for k, v in obj.vocab.items()}
        return obj


class ReasoningTokenizer:
    def __init__(self, tokenizer: AureliusTokenizer) -> None:
        self.tokenizer = tokenizer

    def wrap_reasoning(self, text: str, mode: str = "standard") -> str:
        if mode == "deep":
            return (
                f"<|deep_think|><|plan|>{text}</|plan|>"
                f"<|scratchpad|>{text}</|scratchpad|>"
                f"<|final_answer|>{text}</|final_answer|></|deep_think|>"
            )
        if mode == "verify":
            return (
                f"<|verify|><|self_check|>{text}</|self_check|>"
                f"<|conclusion|>{text}</|conclusion|></|verify|>"
            )
        return f"<|think|>{text}</|think|>"

    def extract_reasoning(self, ids: list[int]) -> dict[str, list[int]]:
        think_open = self.tokenizer.special_tokens["<|think|>"]
        think_close = self.tokenizer.special_tokens["</|think|>"]
        sections = {"think": [], "text": []}
        current = "think"
        for token_id in ids:
            if token_id == think_open:
                current = "think"
                continue
            if token_id == think_close:
                current = "text"
                continue
            sections[current].append(token_id)
        return sections


class ToolCallTokenizer:
    @staticmethod
    def encode_call(name: str, params: dict[str, Any]) -> str:
        pieces = ['<|tool_calls|><|invoke|> name="', name, '">']
        for key, value in params.items():
            if isinstance(value, str):
                value_text = value
                is_string = "true"
            else:
                value_text = json.dumps(value, ensure_ascii=False)
                is_string = "false"
            pieces.extend(
                [
                    '<|parameter|> name="',
                    key,
                    '" string="',
                    is_string,
                    '">',
                    value_text,
                    "</|parameter|>",
                ]
            )
        pieces.append("</|invoke|></|tool_calls|>")
        return "".join(pieces)

    @staticmethod
    def parse_call(text: str) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for invoke_match in _INVOKE_RE.finditer(text):
            name, body = invoke_match.groups()
            params: dict[str, Any] = {}
            for param_match in _PARAM_RE.finditer(body):
                key, is_string, raw_value = param_match.groups()
                value: Any
                if is_string == "true":
                    value = raw_value
                else:
                    raw_value = raw_value.strip()
                    if re.fullmatch(r"-?\d+", raw_value):
                        value = int(raw_value)
                    else:
                        try:
                            value = json.loads(raw_value)
                        except Exception:
                            value = raw_value
                params[key] = value
            calls.append({"name": name, "params": params})
        return calls


class MemoryTokenizer:
    @staticmethod
    def store(key: str, content: str) -> str:
        return f"<|memory_store|> key={key} content={content}"

    @staticmethod
    def retrieve(query: str) -> str:
        return f"<|memory_retrieve|> query={query}"

    @staticmethod
    def reference(session_id: str, item_id: int) -> str:
        return f"<|reference|> session={session_id} item={item_id}"


class AgentRoutingTokenizer:
    @staticmethod
    def delegate(task: str, agent_name: str) -> str:
        return f"<|agent|><|delegate|> agent={agent_name} task={task}"

    @staticmethod
    def respond(agent_name: str, result: str) -> str:
        return f"<|agent_result|> agent={agent_name} result={result}"

    @staticmethod
    def swarm_call(tasks: list[str]) -> str:
        task_text = " | ".join(tasks)
        return f"<|swarm|> tasks={task_text}"

    @staticmethod
    def debate_round(text: str, round_num: int) -> str:
        return f"<|debate|> round_{round_num}: {text}"
