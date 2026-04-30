"""General-purpose output contracts for the unified persona system."""

from __future__ import annotations

from ..unified_persona import OutputContract

GENERAL_QA_CONTRACT = OutputContract(
    name="general_qa",
    schema={
        "type": "object",
        "required": ["answer", "confidence", "sources"],
        "properties": {
            "answer": {"type": "string"},
            "confidence": {"type": "string", "enum": ["high", "medium", "low"]},
            "sources": {"type": "array", "items": {"type": "string"}},
        },
    },
    required_fields=("answer", "confidence"),
)

CODE_REVIEW_CONTRACT = OutputContract(
    name="code_review",
    schema={
        "type": "object",
        "required": ["summary", "issues", "suggestions"],
        "properties": {
            "summary": {"type": "string"},
            "issues": {"type": "array", "items": {"type": "object"}},
            "suggestions": {"type": "array", "items": {"type": "string"}},
        },
    },
    required_fields=("summary", "issues"),
)

EXPLANATION_CONTRACT = OutputContract(
    name="explanation",
    schema={
        "type": "object",
        "required": ["topic", "explanation", "difficulty"],
        "properties": {
            "topic": {"type": "string"},
            "explanation": {"type": "string"},
            "difficulty": {"type": "string", "enum": ["beginner", "intermediate", "advanced"]},
            "prerequisites": {"type": "array", "items": {"type": "string"}},
            "examples": {"type": "array", "items": {"type": "string"}},
        },
    },
    required_fields=("topic", "explanation"),
)

DEBUG_ANALYSIS_CONTRACT = OutputContract(
    name="debug_analysis",
    schema={
        "type": "object",
        "required": ["root_cause", "fix", "steps_to_reproduce"],
        "properties": {
            "root_cause": {"type": "string"},
            "fix": {"type": "string"},
            "steps_to_reproduce": {"type": "array", "items": {"type": "string"}},
            "related_issues": {"type": "array", "items": {"type": "string"}},
        },
    },
    required_fields=("root_cause", "fix", "steps_to_reproduce"),
)

__all__ = [
    "GENERAL_QA_CONTRACT",
    "CODE_REVIEW_CONTRACT",
    "EXPLANATION_CONTRACT",
    "DEBUG_ANALYSIS_CONTRACT",
]
