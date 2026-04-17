"""Tests for src/data/raft_pipeline.py — RAFT data construction pipeline."""

from __future__ import annotations

import pytest

from src.data.raft_pipeline import (
    RAFTConfig,
    RAFTDocument,
    RAFTPipeline,
    RAFTSample,
    compute_oracle_recall,
    extract_answer_from_output,
    mock_documents,
    mock_qa_pairs,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def default_config() -> RAFTConfig:
    return RAFTConfig(
        n_distractor_docs=3,
        n_oracle_docs=1,
        oracle_prob=0.8,
        max_doc_length=500,
    )


@pytest.fixture
def oracle_doc() -> RAFTDocument:
    return RAFTDocument(
        doc_id="oracle_0",
        content="Paris is the capital of France and home to the Eiffel Tower.",
        is_oracle=True,
        source="encyclopedia",
    )


@pytest.fixture
def distractor_docs() -> list[RAFTDocument]:
    return [
        RAFTDocument(
            doc_id=f"dist_{i}",
            content=f"This is an unrelated document about topic {i}.",
            is_oracle=False,
            source=f"source_{i}",
        )
        for i in range(5)
    ]


@pytest.fixture
def pipeline(default_config) -> RAFTPipeline:
    return RAFTPipeline(config=default_config, seed=0)


@pytest.fixture
def sample_always_oracle(oracle_doc, distractor_docs, default_config) -> RAFTSample:
    """Sample built with oracle_prob=1.0 so oracle is always included."""
    cfg = RAFTConfig(
        n_distractor_docs=3,
        oracle_prob=1.0,
        max_doc_length=500,
    )
    pipe = RAFTPipeline(config=cfg, seed=0)
    return pipe.build_sample(
        question="What is the capital of France?",
        answer="Paris",
        oracle_doc=oracle_doc,
        distractor_docs=distractor_docs,
        chain_of_thought="France has a capital called Paris.",
    )


# ---------------------------------------------------------------------------
# 1. RAFTDocument: is_oracle distinguishes oracle from distractors
# ---------------------------------------------------------------------------

def test_raft_document_is_oracle_true(oracle_doc):
    assert oracle_doc.is_oracle is True


def test_raft_document_is_oracle_false(distractor_docs):
    for doc in distractor_docs:
        assert doc.is_oracle is False


# ---------------------------------------------------------------------------
# 2. RAFTSample.format_context has 'Document N:' formatting
# ---------------------------------------------------------------------------

def test_format_context_document_labels(sample_always_oracle):
    context = sample_always_oracle.format_context()
    # There should be at least one "Document N:" label
    assert "Document 1:" in context


def test_format_context_sequential_numbering(sample_always_oracle):
    context = sample_always_oracle.format_context()
    n_docs = len(sample_always_oracle.documents)
    for i in range(1, n_docs + 1):
        assert f"Document {i}:" in context


# ---------------------------------------------------------------------------
# 3. RAFTSample.format_input contains both context and question
# ---------------------------------------------------------------------------

def test_format_input_contains_context(sample_always_oracle):
    fmt = sample_always_oracle.format_input()
    assert "Document 1:" in fmt


def test_format_input_contains_question(sample_always_oracle):
    fmt = sample_always_oracle.format_input()
    assert "What is the capital of France?" in fmt


def test_format_input_question_after_context(sample_always_oracle):
    fmt = sample_always_oracle.format_input()
    ctx_end = fmt.rfind("Document")
    q_start = fmt.find("Question:")
    assert q_start > ctx_end


# ---------------------------------------------------------------------------
# 4. RAFTSample.format_output contains answer_prefix
# ---------------------------------------------------------------------------

def test_format_output_contains_answer_prefix(sample_always_oracle):
    output = sample_always_oracle.format_output()
    assert "##ANSWER:" in output


def test_format_output_contains_answer(sample_always_oracle):
    output = sample_always_oracle.format_output()
    assert "Paris" in output


def test_format_output_contains_cot_prefix(sample_always_oracle):
    output = sample_always_oracle.format_output()
    assert "Let me think step by step." in output


# ---------------------------------------------------------------------------
# 5. RAFTSample.to_sft_example has instruction/input/output keys
# ---------------------------------------------------------------------------

def test_to_sft_example_keys(sample_always_oracle):
    ex = sample_always_oracle.to_sft_example()
    assert "instruction" in ex
    assert "input" in ex
    assert "output" in ex


def test_to_sft_example_values_are_strings(sample_always_oracle):
    ex = sample_always_oracle.to_sft_example()
    for key in ("instruction", "input", "output"):
        assert isinstance(ex[key], str)
        assert len(ex[key]) > 0


# ---------------------------------------------------------------------------
# 6. RAFTPipeline.build_sample returns RAFTSample
# ---------------------------------------------------------------------------

def test_build_sample_returns_raft_sample(pipeline, oracle_doc, distractor_docs):
    sample = pipeline.build_sample(
        question="Q?",
        answer="A",
        oracle_doc=oracle_doc,
        distractor_docs=distractor_docs,
    )
    assert isinstance(sample, RAFTSample)


# ---------------------------------------------------------------------------
# 7. Oracle doc included in sample when oracle_prob=1.0
# ---------------------------------------------------------------------------

def test_oracle_included_when_prob_one(oracle_doc, distractor_docs):
    cfg = RAFTConfig(n_distractor_docs=3, oracle_prob=1.0)
    pipe = RAFTPipeline(config=cfg, seed=42)
    sample = pipe.build_sample(
        question="Q?",
        answer="A",
        oracle_doc=oracle_doc,
        distractor_docs=distractor_docs,
    )
    doc_ids = [d.doc_id for d in sample.documents]
    assert oracle_doc.doc_id in doc_ids


def test_oracle_doc_id_set_when_included(oracle_doc, distractor_docs):
    cfg = RAFTConfig(n_distractor_docs=3, oracle_prob=1.0)
    pipe = RAFTPipeline(config=cfg, seed=42)
    sample = pipe.build_sample(
        question="Q?",
        answer="A",
        oracle_doc=oracle_doc,
        distractor_docs=distractor_docs,
    )
    assert sample.oracle_doc_id == oracle_doc.doc_id


# ---------------------------------------------------------------------------
# 8. No oracle doc included when oracle_prob=0.0
# ---------------------------------------------------------------------------

def test_oracle_excluded_when_prob_zero(oracle_doc, distractor_docs):
    cfg = RAFTConfig(n_distractor_docs=3, oracle_prob=0.0)
    pipe = RAFTPipeline(config=cfg, seed=42)
    sample = pipe.build_sample(
        question="Q?",
        answer="A",
        oracle_doc=oracle_doc,
        distractor_docs=distractor_docs,
    )
    doc_ids = [d.doc_id for d in sample.documents]
    assert oracle_doc.doc_id not in doc_ids


def test_oracle_doc_id_empty_when_excluded(oracle_doc, distractor_docs):
    cfg = RAFTConfig(n_distractor_docs=3, oracle_prob=0.0)
    pipe = RAFTPipeline(config=cfg, seed=42)
    sample = pipe.build_sample(
        question="Q?",
        answer="A",
        oracle_doc=oracle_doc,
        distractor_docs=distractor_docs,
    )
    assert sample.oracle_doc_id == ""


# ---------------------------------------------------------------------------
# 9. Total docs = n_oracle_docs + n_distractor_docs when oracle is included
# ---------------------------------------------------------------------------

def test_total_docs_count_with_oracle(oracle_doc, distractor_docs):
    cfg = RAFTConfig(n_distractor_docs=3, n_oracle_docs=1, oracle_prob=1.0)
    pipe = RAFTPipeline(config=cfg, seed=42)
    sample = pipe.build_sample(
        question="Q?",
        answer="A",
        oracle_doc=oracle_doc,
        distractor_docs=distractor_docs,
    )
    expected = cfg.n_oracle_docs + cfg.n_distractor_docs
    assert len(sample.documents) == expected


def test_total_docs_count_without_oracle(oracle_doc, distractor_docs):
    cfg = RAFTConfig(n_distractor_docs=3, n_oracle_docs=1, oracle_prob=0.0)
    pipe = RAFTPipeline(config=cfg, seed=42)
    sample = pipe.build_sample(
        question="Q?",
        answer="A",
        oracle_doc=oracle_doc,
        distractor_docs=distractor_docs,
    )
    # Only distractors
    assert len(sample.documents) == cfg.n_distractor_docs


# ---------------------------------------------------------------------------
# 10. create_dataset returns list of RAFTSample
# ---------------------------------------------------------------------------

def test_create_dataset_returns_list_of_raft_samples(pipeline):
    docs = mock_documents(8)
    qa = mock_qa_pairs(4)
    oracle_assignments = [docs[0].doc_id] * 4
    samples = pipeline.create_dataset(qa, docs, oracle_assignments)
    assert isinstance(samples, list)
    assert len(samples) == 4
    for s in samples:
        assert isinstance(s, RAFTSample)


def test_create_dataset_uses_cot_generator(pipeline):
    docs = mock_documents(8)
    qa = mock_qa_pairs(2)
    oracle_assignments = [docs[0].doc_id] * 2

    def my_cot(question, answer, doc):
        return f"Based on {doc.doc_id}, the answer is {answer}."

    samples = pipeline.create_dataset(qa, docs, oracle_assignments, cot_generator=my_cot)
    for s in samples:
        assert s.chain_of_thought != ""


# ---------------------------------------------------------------------------
# 11. filter_high_quality removes samples with short documents
# ---------------------------------------------------------------------------

def test_filter_removes_short_docs(pipeline):
    short_doc = RAFTDocument(doc_id="short", content="Hi", is_oracle=False)
    long_doc = RAFTDocument(
        doc_id="long",
        content="A" * 100,
        is_oracle=True,
    )
    cfg_always = RAFTConfig(n_distractor_docs=1, oracle_prob=1.0)
    pipe = RAFTPipeline(config=cfg_always, seed=0)

    good_sample = pipe.build_sample(
        question="Q?",
        answer="A",
        oracle_doc=long_doc,
        distractor_docs=[RAFTDocument(doc_id="d", content="B" * 100, is_oracle=False)],
    )
    bad_sample = RAFTSample(
        question="Q?",
        documents=[short_doc],
        answer="A",
        _config=cfg_always,
    )

    filtered = pipeline.filter_high_quality([good_sample, bad_sample], min_doc_length=50)
    assert good_sample in filtered
    assert bad_sample not in filtered


def test_filter_require_cot(pipeline, oracle_doc, distractor_docs):
    cfg_always = RAFTConfig(n_distractor_docs=3, oracle_prob=1.0)
    pipe = RAFTPipeline(config=cfg_always, seed=0)

    with_cot = pipe.build_sample("Q?", "A", oracle_doc, distractor_docs, chain_of_thought="Here is my reasoning.")
    without_cot = pipe.build_sample("Q?", "A", oracle_doc, distractor_docs, chain_of_thought="")

    filtered = pipeline.filter_high_quality([with_cot, without_cot], min_doc_length=10, require_cot=True)
    assert with_cot in filtered
    assert without_cot not in filtered


# ---------------------------------------------------------------------------
# 12. extract_answer_from_output correctly extracts answer
# ---------------------------------------------------------------------------

def test_extract_answer_basic():
    output = "Let me think step by step.\nSome reasoning.\n##ANSWER: Paris"
    assert extract_answer_from_output(output) == "Paris"


def test_extract_answer_multi_word():
    output = "Thoughts here.\n##ANSWER: William Shakespeare"
    assert extract_answer_from_output(output) == "William Shakespeare"


def test_extract_answer_missing_prefix():
    output = "No answer prefix here."
    assert extract_answer_from_output(output) == ""


def test_extract_answer_custom_prefix():
    output = "Blah blah ##ANS: 42"
    assert extract_answer_from_output(output, answer_prefix="##ANS:") == "42"


# ---------------------------------------------------------------------------
# 13. compute_oracle_recall returns dict with answer_present, oracle_cited
# ---------------------------------------------------------------------------

def test_compute_oracle_recall_keys(sample_always_oracle):
    result = compute_oracle_recall(sample_always_oracle, "##ANSWER: Paris oracle_0")
    assert "answer_present" in result
    assert "oracle_cited" in result


def test_compute_oracle_recall_answer_present(sample_always_oracle):
    result = compute_oracle_recall(sample_always_oracle, "##ANSWER: Paris")
    assert result["answer_present"] == 1.0


def test_compute_oracle_recall_answer_absent(sample_always_oracle):
    result = compute_oracle_recall(sample_always_oracle, "##ANSWER: Berlin")
    assert result["answer_present"] == 0.0


def test_compute_oracle_recall_oracle_cited(sample_always_oracle):
    result = compute_oracle_recall(sample_always_oracle, f"See oracle_0. ##ANSWER: Paris")
    assert result["oracle_cited"] == 1.0


def test_compute_oracle_recall_oracle_not_cited(sample_always_oracle):
    result = compute_oracle_recall(sample_always_oracle, "##ANSWER: Paris")
    assert result["oracle_cited"] == 0.0


# ---------------------------------------------------------------------------
# 14. mock_documents generates a mix of oracle and non-oracle docs
# ---------------------------------------------------------------------------

def test_mock_documents_count():
    docs = mock_documents(8)
    assert len(docs) == 8


def test_mock_documents_has_oracle():
    docs = mock_documents(8)
    oracles = [d for d in docs if d.is_oracle]
    assert len(oracles) >= 1


def test_mock_documents_has_distractors():
    docs = mock_documents(8)
    distractors = [d for d in docs if not d.is_oracle]
    assert len(distractors) >= 1


def test_mock_documents_unique_ids():
    docs = mock_documents(8)
    ids = [d.doc_id for d in docs]
    assert len(set(ids)) == len(ids)


def test_mock_qa_pairs_count():
    pairs = mock_qa_pairs(4)
    assert len(pairs) == 4
    for q, a in pairs:
        assert isinstance(q, str) and isinstance(a, str)
