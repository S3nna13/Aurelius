"""Kaggle NLP dataset loaders for Aurelius training pipeline.

Supports CommonLit Readability, Jigsaw Toxic Comment Classification,
Google QUEST Q&A Labeling, and Feedback Prize discourse element formats.
Pure Python, no network or torch dependencies.
"""
from __future__ import annotations

from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Label constants
# ---------------------------------------------------------------------------

TOXICITY_LABEL_NAMES: list[str] = [
    "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"
]

DISCOURSE_TYPES: set[str] = {
    "Lead", "Position", "Claim", "Evidence",
    "Concluding Statement", "Counterclaim", "Rebuttal",
}

# All 30 QUEST float label column names (subset used for mock; full list for parsing)
QUEST_LABEL_COLUMNS: list[str] = [
    "question_asker_intent_understanding",
    "question_body_critical",
    "question_conversational",
    "question_expect_short_answer",
    "question_fact_seeking",
    "question_has_commonly_accepted_answer",
    "question_interestingness_others",
    "question_interestingness_self",
    "question_multi_intent",
    "question_not_really_a_question",
    "question_opinion_seeking",
    "question_type_choice",
    "question_type_compare",
    "question_type_consequence",
    "question_type_definition",
    "question_type_entity",
    "question_type_instructions",
    "question_type_procedure",
    "question_type_reason_explanation",
    "question_type_spelling",
    "question_well_written",
    "answer_helpful",
    "answer_level_of_information",
    "answer_plausible",
    "answer_relevance",
    "answer_satisfaction",
    "answer_type_instructions",
    "answer_type_procedure",
    "answer_type_reason_explanation",
    "answer_well_written",
]


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------

@dataclass
class JigsawSample:
    """A single Jigsaw Toxic Comment Classification sample."""
    id: str
    text: str
    toxic: int
    severe_toxic: int
    obscene: int
    threat: int
    insult: int
    identity_hate: int

    @property
    def any_toxic(self) -> bool:
        """True if any toxicity label is set."""
        return any([
            self.toxic, self.severe_toxic, self.obscene,
            self.threat, self.insult, self.identity_hate,
        ])

    @property
    def toxicity_labels(self) -> list[str]:
        """Return list of active toxicity label names."""
        values = {
            "toxic": self.toxic,
            "severe_toxic": self.severe_toxic,
            "obscene": self.obscene,
            "threat": self.threat,
            "insult": self.insult,
            "identity_hate": self.identity_hate,
        }
        return [name for name, val in values.items() if val]


@dataclass
class QALabelSample:
    """A single Google QUEST Q&A Labeling sample."""
    qa_id: str
    question_title: str
    question_body: str
    answer: str
    category: str
    labels: dict[str, float] = field(default_factory=dict)  # 30 float labels


@dataclass
class DiscourseElement:
    """A single Feedback Prize discourse element."""
    id: str
    discourse_id: str
    text: str
    discourse_type: str  # Lead/Position/Claim/Evidence/etc.
    start: int
    end: int


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------

def parse_jigsaw_sample(row: dict) -> JigsawSample:
    """Parse a raw Jigsaw CSV row dict into a JigsawSample."""
    return JigsawSample(
        id=str(row.get("id", "")),
        text=str(row.get("comment_text", "")),
        toxic=int(row.get("toxic", 0)),
        severe_toxic=int(row.get("severe_toxic", 0)),
        obscene=int(row.get("obscene", 0)),
        threat=int(row.get("threat", 0)),
        insult=int(row.get("insult", 0)),
        identity_hate=int(row.get("identity_hate", 0)),
    )


def parse_qa_sample(row: dict) -> QALabelSample:
    """Parse a raw QUEST Q&A CSV row dict into a QALabelSample."""
    labels: dict[str, float] = {}
    for col in QUEST_LABEL_COLUMNS:
        if col in row:
            try:
                labels[col] = float(row[col])
            except (TypeError, ValueError):
                labels[col] = 0.0
    return QALabelSample(
        qa_id=str(row.get("qa_id", "")),
        question_title=str(row.get("question_title", "")),
        question_body=str(row.get("question_body", "")),
        answer=str(row.get("answer", "")),
        category=str(row.get("category", "")),
        labels=labels,
    )


def parse_discourse_element(row: dict) -> DiscourseElement:
    """Parse a raw Feedback Prize CSV row dict into a DiscourseElement."""
    return DiscourseElement(
        id=str(row.get("id", "")),
        discourse_id=str(row.get("discourse_id", "")),
        text=str(row.get("discourse_text", "")),
        discourse_type=str(row.get("discourse_type", "")),
        start=int(row.get("discourse_start", 0)),
        end=int(row.get("discourse_end", 0)),
    )


# ---------------------------------------------------------------------------
# Instruction formatters
# ---------------------------------------------------------------------------

def jigsaw_to_instruction(sample: JigsawSample) -> dict:
    """Convert a JigsawSample to an instruction-tuning dict."""
    return {
        "instruction": "Classify if this comment is toxic",
        "input": sample.text,
        "output": "toxic" if sample.any_toxic else "non-toxic",
    }


def qa_to_instruction(sample: QALabelSample) -> dict:
    """Convert a QALabelSample to an instruction-tuning dict."""
    return {
        "instruction": "Answer this question",
        "input": f"Q: {sample.question_title}\n{sample.question_body}",
        "output": sample.answer,
    }


def discourse_to_instruction(element: DiscourseElement) -> dict:
    """Convert a DiscourseElement to an instruction-tuning dict."""
    return {
        "instruction": "Classify this discourse element",
        "input": element.text,
        "output": element.discourse_type,
    }


# ---------------------------------------------------------------------------
# Filters
# ---------------------------------------------------------------------------

def filter_toxic(
    samples: list[JigsawSample],
    min_label: str = "toxic",
) -> list[JigsawSample]:
    """Filter samples where the specified toxicity label == 1.

    If min_label is not a recognised toxicity label, returns an empty list
    rather than raising an exception.
    """
    if min_label not in TOXICITY_LABEL_NAMES:
        return []
    return [s for s in samples if getattr(s, min_label, 0) == 1]


# ---------------------------------------------------------------------------
# Mock data generators
# ---------------------------------------------------------------------------

def mock_jigsaw_data(n: int = 4) -> list[dict]:
    """Generate n synthetic Jigsaw CSV row dicts for testing."""
    rows = []
    for i in range(n):
        rows.append({
            "id": f"jig_{i:04d}",
            "comment_text": f"This is comment number {i}.",
            "toxic": i % 2,          # alternates 0/1
            "severe_toxic": 0,
            "obscene": i % 3 == 0,   # every third
            "threat": 0,
            "insult": 0,
            "identity_hate": 0,
        })
    return rows


def mock_qa_data(n: int = 4) -> list[dict]:
    """Generate n synthetic QUEST Q&A CSV row dicts for testing."""
    rows = []
    for i in range(n):
        base: dict = {
            "qa_id": f"qa_{i:04d}",
            "question_title": f"What is concept {i}?",
            "question_body": f"I am trying to understand concept {i} in detail.",
            "question_user_name": f"user_{i}",
            "answer": f"Concept {i} refers to a fundamental principle.",
            "answer_user_name": f"expert_{i}",
            "category": "SCIENCE" if i % 2 == 0 else "TECHNOLOGY",
            "host": "stackoverflow.com",
            "url": f"https://stackoverflow.com/questions/{i}",
        }
        # Add 30 float label columns
        for col in QUEST_LABEL_COLUMNS:
            base[col] = round(0.1 * (i % 10), 4)
        rows.append(base)
    return rows


def mock_discourse_data(n: int = 4) -> list[dict]:
    """Generate n synthetic Feedback Prize CSV row dicts for testing."""
    types = list(DISCOURSE_TYPES)
    rows = []
    for i in range(n):
        dtype = types[i % len(types)]
        start = i * 100
        end = start + 50
        rows.append({
            "id": f"essay_{i:04d}",
            "discourse_id": f"disc_{i:06d}",
            "discourse_start": start,
            "discourse_end": end,
            "discourse_text": f"This is the {dtype.lower()} discourse text {i}.",
            "discourse_type": dtype,
            "discourse_type_num": i % len(types),
            "predictionstring": f"{start} {start+1} {start+2}",
        })
    return rows
