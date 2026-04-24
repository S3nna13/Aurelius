"""Magpie synthetic instruction generator.

Adapted from the Heavens_Gate Magpie generator for Aurelius. Produces
(instruction, response) pairs by sampling an aligned LLM with only its
pre-query template header, then parsing the self-generated conversation.

Reference: Magpie (ICLR 2025) — https://github.com/magpie-align/magpie

Stdlib only: ``json``, ``uuid``, ``pathlib``, ``dataclasses``.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from .domain_templates import DomainTemplate


@dataclass
class MagpieConfig:
    """Runtime configuration for :class:`MagpieGenerator`."""

    model_id: str = "dummy"
    output_dir: str = "data/synthetic"
    batch_size: int = 32
    max_new_tokens: int = 2048
    temperature: float = 0.9
    seed: int = 42


@dataclass(frozen=True)
class MagpieExample:
    """A single Magpie (instruction, response) pair with domain metadata."""

    instruction: str
    response: str
    domain: str
    template_prefix: str = ""


GenerateFn = Callable[..., str]

_USER_RE = re.compile(r"^\s*(user|human)\s*:\s*", re.IGNORECASE)
_ASSISTANT_RE = re.compile(r"(?i)\bassistant\s*:\s*")


def _split_text(text: str) -> tuple[str, str]:
    """Parse ``User:``/``Assistant:`` conversation text into (instruction, response).

    Falls back to splitting at the first blank line when no explicit markers
    are present.
    """
    if not text:
        return "", ""

    match = _ASSISTANT_RE.search(text)
    if match:
        head = text[: match.start()]
        tail = text[match.end() :]
        instruction = _USER_RE.sub("", head).strip()
        response = tail.strip()
        if instruction and response:
            return instruction, response

    halves = text.split("\n\n", 1)
    if len(halves) == 2:
        instruction = _USER_RE.sub("", halves[0]).strip()
        response = halves[1].strip()
        return instruction, response

    return "", ""


class MagpieGenerator:
    """Magpie-style instruction generator driven by a caller-supplied LLM fn."""

    def __init__(self, config: MagpieConfig) -> None:
        self.config = config

    def generate_from_template(
        self,
        template: DomainTemplate,
        n: int,
        generate_fn: GenerateFn,
    ) -> list[MagpieExample]:
        """Generate *n* examples by repeatedly calling *generate_fn*.

        ``generate_fn`` is invoked as
        ``generate_fn(prompt=..., max_tokens=..., temperature=...) -> str``.
        """
        if n < 0:
            raise ValueError("n must be non-negative")

        examples: list[MagpieExample] = []
        for _ in range(n):
            raw = generate_fn(
                prompt=template.prefix,
                max_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
            )
            instruction, response = _split_text(raw or "")
            if not instruction or not response:
                continue
            examples.append(
                MagpieExample(
                    instruction=instruction,
                    response=response,
                    domain=template.domain,
                    template_prefix=template.prefix,
                )
            )
        return examples

    def export_jsonl(self, examples: list[MagpieExample], path: str) -> int:
        """Write *examples* as JSONL to *path*. Returns the number of records."""
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as fh:
            for ex in examples:
                record = {
                    "instruction": ex.instruction,
                    "response": ex.response,
                    "domain": ex.domain,
                    "template_prefix": ex.template_prefix,
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
        return len(examples)

    def load_jsonl(self, path: str) -> list[MagpieExample]:
        """Read a Magpie JSONL file back into a list of :class:`MagpieExample`."""
        examples: list[MagpieExample] = []
        with Path(path).open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                examples.append(
                    MagpieExample(
                        instruction=record.get("instruction", ""),
                        response=record.get("response", ""),
                        domain=record.get("domain", ""),
                        template_prefix=record.get("template_prefix", ""),
                    )
                )
        return examples

    def stats(self, examples: list[MagpieExample]) -> dict:
        """Return aggregate counts and average lengths for *examples*."""
        total = len(examples)
        by_domain: dict[str, int] = {}
        instr_total = 0
        resp_total = 0
        for ex in examples:
            by_domain[ex.domain] = by_domain.get(ex.domain, 0) + 1
            instr_total += len(ex.instruction)
            resp_total += len(ex.response)
        avg_i = (instr_total / total) if total else 0.0
        avg_r = (resp_total / total) if total else 0.0
        return {
            "total": total,
            "by_domain": by_domain,
            "avg_instruction_len": avg_i,
            "avg_response_len": avg_r,
        }


MAGPIE_GENERATOR_REGISTRY = {"default": MagpieGenerator}
