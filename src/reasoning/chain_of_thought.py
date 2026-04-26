"""Chain-of-thought: step extractor, consistency checker, format templates."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class CoTFormat(StrEnum):
    NUMBERED = "numbered"
    BULLET = "bullet"
    XML_TAGS = "xml_tags"
    FREEFORM = "freeform"


@dataclass
class CoTStep:
    index: int
    content: str
    confidence: float = 1.0


class ChainOfThought:
    def __init__(self, format: CoTFormat = CoTFormat.NUMBERED) -> None:
        self.format = format

    def parse_steps(self, text: str) -> list[CoTStep]:
        steps: list[CoTStep] = []
        if not text or not text.strip():
            return steps

        if self.format == CoTFormat.NUMBERED:
            for line in text.splitlines():
                m = re.match(r"^\s*(\d+)[.)]\s+(.+)", line)
                if m:
                    steps.append(CoTStep(index=int(m.group(1)), content=m.group(2).strip()))

        elif self.format == CoTFormat.BULLET:
            line_number = 0
            for line in text.splitlines():
                m = re.match(r"^\s*[-*•]\s+(.+)", line)
                if m:
                    steps.append(CoTStep(index=line_number, content=m.group(1).strip()))
                    line_number += 1

        elif self.format == CoTFormat.XML_TAGS:
            matches = re.findall(r"<step>(.*?)</step>", text, re.DOTALL)
            for i, content in enumerate(matches):
                steps.append(CoTStep(index=i, content=content.strip()))

        elif self.format == CoTFormat.FREEFORM:
            paragraphs = re.split(r"\n\n+", text.strip())
            for i, para in enumerate(paragraphs):
                para = para.strip()
                if para:
                    steps.append(CoTStep(index=i, content=para))

        return steps

    def format_prompt(self, question: str) -> str:
        hints = {
            CoTFormat.NUMBERED: "Think step by step. Number each step (e.g., 1. ... 2. ...).",
            CoTFormat.BULLET: "Think step by step. Use bullet points (- or *) for each step.",
            CoTFormat.XML_TAGS: "Think step by step. Wrap each step in <step>...</step> tags.",
            CoTFormat.FREEFORM: "Think step by step. Separate each step with a blank line.",
        }
        hint = hints[self.format]
        return f"{hint}\n\nQuestion: {question}"

    def consistency_score(self, steps: list[CoTStep]) -> float:
        if not steps:
            return 1.0
        return sum(s.confidence for s in steps) / len(steps)

    def to_text(self, steps: list[CoTStep]) -> str:
        lines: list[str] = []
        if self.format == CoTFormat.NUMBERED:
            for s in steps:
                lines.append(f"{s.index}. {s.content}")
        elif self.format == CoTFormat.BULLET:
            for s in steps:
                lines.append(f"- {s.content}")
        elif self.format == CoTFormat.XML_TAGS:
            for s in steps:
                lines.append(f"<step>{s.content}</step>")
        elif self.format == CoTFormat.FREEFORM:
            lines = [s.content for s in steps]
            return "\n\n".join(lines)
        return "\n".join(lines)


COT_REGISTRY: dict[str, ChainOfThought] = {
    "numbered": ChainOfThought(CoTFormat.NUMBERED),
    "bullet": ChainOfThought(CoTFormat.BULLET),
    "xml": ChainOfThought(CoTFormat.XML_TAGS),
}
