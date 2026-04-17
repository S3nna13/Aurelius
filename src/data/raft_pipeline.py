# src/data/raft_pipeline.py
"""RAFT: Adapting Language Model to Domain Specific RAG (Zhang et al. 2024).

Constructs RAFT training data from QA pairs + document corpora, teaching the
model to identify relevant oracle documents, ignore distractors, and cite
sources while producing chain-of-thought answers.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass
class RAFTConfig:
    """Configuration for the RAFT data-construction pipeline."""

    n_distractor_docs: int = 3    # number of distractor (irrelevant) documents
    n_oracle_docs: int = 1        # number of relevant (oracle) documents
    oracle_prob: float = 0.8      # probability of including oracle doc in context
    max_doc_length: int = 200     # max characters per document (truncation guard)
    cot_prefix: str = "Let me think step by step."
    answer_prefix: str = "##ANSWER:"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RAFTDocument:
    """A single document in the RAFT corpus."""

    doc_id: str
    content: str
    is_oracle: bool = False   # True = this document contains the answer
    source: str = ""


@dataclass
class RAFTSample:
    """A single RAFT training example."""

    question: str
    documents: list[RAFTDocument]   # oracle + distractors, interleaved
    answer: str
    chain_of_thought: str = ""
    oracle_doc_id: str = ""         # doc_id of the oracle document
    # back-reference to config so format_output knows the prefixes
    _config: RAFTConfig = field(default_factory=RAFTConfig, repr=False, compare=False)

    # ------------------------------------------------------------------
    # Formatting helpers
    # ------------------------------------------------------------------

    def format_context(self) -> str:
        """Return documents formatted as 'Document N: {content}' blocks."""
        parts = []
        for i, doc in enumerate(self.documents, start=1):
            content = doc.content[: self._config.max_doc_length]
            parts.append(f"Document {i}: {content}")
        return "\n".join(parts)

    def format_input(self) -> str:
        """Return '{context}\\n\\nQuestion: {question}'."""
        context = self.format_context()
        return f"{context}\n\nQuestion: {self.question}"

    def format_output(self) -> str:
        """Return '{cot_prefix}\\n{chain_of_thought}\\n{answer_prefix} {answer}'."""
        cot = self.chain_of_thought or ""
        return (
            f"{self._config.cot_prefix}\n"
            f"{cot}\n"
            f"{self._config.answer_prefix} {self.answer}"
        )

    def to_sft_example(self) -> dict:
        """Return a supervised fine-tuning dict with instruction/input/output keys."""
        return {
            "instruction": (
                "Answer the question using the provided documents. "
                "Cite the relevant document and ignore distractors."
            ),
            "input": self.format_input(),
            "output": self.format_output(),
        }


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

class RAFTPipeline:
    """Pipeline for constructing RAFT training data from QA pairs + document corpus."""

    def __init__(self, config: RAFTConfig | None = None, seed: int = 42) -> None:
        self.config = config or RAFTConfig()
        self.rng = random.Random(seed)

    # ------------------------------------------------------------------
    # Core builder
    # ------------------------------------------------------------------

    def build_sample(
        self,
        question: str,
        answer: str,
        oracle_doc: RAFTDocument,
        distractor_docs: list[RAFTDocument],
        chain_of_thought: str = "",
    ) -> RAFTSample:
        """Construct a single RAFT sample.

        With probability ``oracle_prob`` the oracle document is included.
        Always includes up to ``n_distractor_docs`` distractors. Documents are
        shuffled so the model cannot rely on position to find the answer.
        """
        cfg = self.config

        # Decide whether to include the oracle document.
        include_oracle = self.rng.random() < cfg.oracle_prob

        # Sample distractor pool.
        available_distractors = [d for d in distractor_docs if d.doc_id != oracle_doc.doc_id]
        n_distractors = min(cfg.n_distractor_docs, len(available_distractors))
        chosen_distractors = self.rng.sample(available_distractors, n_distractors)

        # Build document list.
        docs: list[RAFTDocument] = []
        if include_oracle:
            docs.append(oracle_doc)
        docs.extend(chosen_distractors)

        # Shuffle so oracle position is random.
        self.rng.shuffle(docs)

        return RAFTSample(
            question=question,
            documents=docs,
            answer=answer,
            chain_of_thought=chain_of_thought,
            oracle_doc_id=oracle_doc.doc_id if include_oracle else "",
            _config=cfg,
        )

    # ------------------------------------------------------------------
    # Dataset creation
    # ------------------------------------------------------------------

    def create_dataset(
        self,
        qa_pairs: list[tuple[str, str]],
        documents: list[RAFTDocument],
        oracle_assignments: list[str],
        cot_generator: callable = None,
    ) -> list[RAFTSample]:
        """Create a full RAFT dataset from QA pairs and a document corpus.

        Args:
            qa_pairs: List of (question, answer) tuples.
            documents: Full document corpus.
            oracle_assignments: Parallel list mapping each qa_pair to its oracle doc_id.
            cot_generator: Optional callable(question, answer, oracle_doc) -> str
                           that produces a chain-of-thought string.

        Returns:
            List of RAFTSample instances.
        """
        doc_index: dict[str, RAFTDocument] = {d.doc_id: d for d in documents}
        samples: list[RAFTSample] = []

        for (question, answer), oracle_id in zip(qa_pairs, oracle_assignments):
            oracle_doc = doc_index.get(oracle_id)
            if oracle_doc is None:
                continue  # skip if oracle doc not in corpus

            distractors = [d for d in documents if d.doc_id != oracle_id]

            cot = ""
            if cot_generator is not None:
                cot = cot_generator(question, answer, oracle_doc)

            sample = self.build_sample(
                question=question,
                answer=answer,
                oracle_doc=oracle_doc,
                distractor_docs=distractors,
                chain_of_thought=cot,
            )
            samples.append(sample)

        return samples

    # ------------------------------------------------------------------
    # Quality filtering
    # ------------------------------------------------------------------

    def filter_high_quality(
        self,
        samples: list[RAFTSample],
        min_doc_length: int = 50,
        require_cot: bool = False,
    ) -> list[RAFTSample]:
        """Filter samples by quality criteria.

        Args:
            samples: List of RAFTSample instances to filter.
            min_doc_length: Minimum character length for ALL documents in a sample.
            require_cot: If True, drop samples that have no chain-of-thought text.

        Returns:
            Filtered list of RAFTSample instances.
        """
        result: list[RAFTSample] = []
        for sample in samples:
            # All documents must meet the minimum length requirement.
            if any(len(doc.content) < min_doc_length for doc in sample.documents):
                continue
            # Optionally require chain-of-thought.
            if require_cot and not sample.chain_of_thought.strip():
                continue
            result.append(sample)
        return result


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def extract_answer_from_output(output: str, answer_prefix: str = "##ANSWER:") -> str:
    """Extract the final answer from a RAFT-formatted output string.

    Finds the last occurrence of ``answer_prefix`` and returns everything after it,
    stripped of leading/trailing whitespace.  Returns an empty string if the prefix
    is not present.
    """
    idx = output.rfind(answer_prefix)
    if idx == -1:
        return ""
    after = output[idx + len(answer_prefix):]
    return after.strip()


def compute_oracle_recall(
    sample: RAFTSample,
    model_output: str,
    answer_prefix: str = "##ANSWER:",
) -> dict[str, float]:
    """Check whether model output contains the answer and cites the oracle document.

    Returns:
        dict with keys:
          - ``answer_present``: 1.0 if the ground-truth answer appears in the output.
          - ``oracle_cited``: 1.0 if the oracle doc_id appears in the output (crude
            citation check). 0.0 if there is no oracle doc in this sample.
    """
    extracted = extract_answer_from_output(model_output, answer_prefix)
    answer_present = 1.0 if sample.answer.lower() in model_output.lower() else 0.0

    oracle_cited = 0.0
    if sample.oracle_doc_id:
        oracle_cited = 1.0 if sample.oracle_doc_id in model_output else 0.0

    return {
        "answer_present": answer_present,
        "oracle_cited": oracle_cited,
    }


# ---------------------------------------------------------------------------
# Mock data generators (for testing / demos)
# ---------------------------------------------------------------------------

def mock_qa_pairs(n: int = 4) -> list[tuple[str, str]]:
    """Return ``n`` synthetic (question, answer) tuples."""
    templates = [
        ("What is the capital of France?", "Paris"),
        ("Who wrote Hamlet?", "William Shakespeare"),
        ("What is the boiling point of water?", "100 degrees Celsius"),
        ("What year did World War II end?", "1945"),
        ("What is the speed of light?", "approximately 299,792 km/s"),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci"),
    ]
    pairs = []
    for i in range(n):
        q, a = templates[i % len(templates)]
        pairs.append((q, a))
    return pairs


def mock_documents(n: int = 8) -> list[RAFTDocument]:
    """Return ``n`` synthetic RAFTDocument objects.

    The first document is always an oracle (is_oracle=True). The rest are
    distractors, giving a realistic mix.
    """
    docs: list[RAFTDocument] = []
    for i in range(n):
        is_oracle = i == 0
        content = (
            f"This is document {i}. "
            + ("It contains the answer: Paris is the capital of France. " * 3
               if is_oracle
               else f"This document discusses topic {i} which is unrelated to the query. " * 3)
        )
        docs.append(
            RAFTDocument(
                doc_id=f"doc_{i}",
                content=content,
                is_oracle=is_oracle,
                source=f"source_{i}",
            )
        )
    return docs
