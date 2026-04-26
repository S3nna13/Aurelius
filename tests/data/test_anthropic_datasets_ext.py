"""Tests for anthropic_datasets_ext.py — all use mock data, no network requests."""

from __future__ import annotations

import base64
from itertools import cycle

from src.data.anthropic_datasets_ext import (
    AlignmentFakingDataset,
    # alignment faking
    AlignmentFakingSample,
    # election questions
    ElectionQuestionSample,
    ElectionQuestionsDataset,
    # global opinions
    GlobalOpinionSample,
    GlobalOpinionsDataset,
    InterviewDataset,
    # interviews
    InterviewTranscript,
    # values-in-the-wild
    ValueFrequency,
    ValuesFrequencyDataset,
    ValuesTreeDataset,
    ValueTreeNode,
    compute_opinion_kl_divergence,
    decrypt_alignment_faking_response,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_KEY = b"MO_transcripts_should_not_appear_in_pretraining_corpora"


def _encrypt(text: str) -> str:
    """Inverse of decrypt_alignment_faking_response (no canary prefix)."""
    return base64.b64encode(bytes(a ^ b for a, b in zip(text.encode(), cycle(_KEY)))).decode()


# ---------------------------------------------------------------------------
# GlobalOpinions
# ---------------------------------------------------------------------------


def _make_global_opinions() -> GlobalOpinionsDataset:
    samples = [
        GlobalOpinionSample(
            question="Do you support X?",
            options=["Yes", "No", "Unsure"],
            country_distributions={"United States": [0.6, 0.3, 0.1]},
            source="GAS",
        ),
        GlobalOpinionSample(
            question="Should Y be allowed?",
            options=["Strongly agree", "Agree", "Disagree"],
            country_distributions={"Germany": [0.2, 0.5, 0.3]},
            source="WVS",
        ),
    ]
    return GlobalOpinionsDataset(samples)


def test_global_opinions_dataset_len():
    ds = _make_global_opinions()
    assert len(ds) == 2


def test_compute_opinion_kl_divergence_same_dist():
    sample = GlobalOpinionSample(
        question="Q",
        options=["A", "B", "C"],
        country_distributions={"United States": [0.5, 0.3, 0.2]},
        source="GAS",
    )
    kl = compute_opinion_kl_divergence(sample, [0.5, 0.3, 0.2], country="United States")
    assert abs(kl) < 1e-9


def test_compute_opinion_kl_divergence_positive():
    sample = GlobalOpinionSample(
        question="Q",
        options=["A", "B"],
        country_distributions={"United States": [0.9, 0.1]},
        source="GAS",
    )
    kl = compute_opinion_kl_divergence(sample, [0.5, 0.5], country="United States")
    assert kl > 0


# ---------------------------------------------------------------------------
# ValuesFrequency
# ---------------------------------------------------------------------------


def _make_values_freq() -> ValuesFrequencyDataset:
    values = [
        ValueFrequency(value="honesty", pct_convos=0.4),
        ValueFrequency(value="helpfulness", pct_convos=0.35),
        ValueFrequency(value="safety", pct_convos=0.25),
    ]
    return ValuesFrequencyDataset(values)


def test_values_frequency_dataset_len():
    ds = _make_values_freq()
    assert len(ds) == 3


def test_values_frequency_weights_sum_to_one():
    ds = _make_values_freq()
    weights = ds.get_weights(temperature=1.0)
    assert abs(sum(weights) - 1.0) < 1e-6
    # Also check with different temperature
    weights_hot = ds.get_weights(temperature=2.0)
    assert abs(sum(weights_hot) - 1.0) < 1e-6


# ---------------------------------------------------------------------------
# ValuesTree
# ---------------------------------------------------------------------------


def _make_values_tree() -> ValuesTreeDataset:
    nodes = [
        ValueTreeNode(
            cluster_id="cat-1",
            name="Ethics",
            level=3,
            parent_cluster_id="",
            pct_total_occurrences=0.4,
            description="Top-level ethics",
        ),
        ValueTreeNode(
            cluster_id="sub-1",
            name="Honesty",
            level=2,
            parent_cluster_id="cat-1",
            pct_total_occurrences=0.2,
            description="Honesty cluster",
        ),
        ValueTreeNode(
            cluster_id="sub-2",
            name="Integrity",
            level=2,
            parent_cluster_id="cat-1",
            pct_total_occurrences=0.2,
            description="Integrity cluster",
        ),
        ValueTreeNode(
            cluster_id="leaf-1",
            name="truth-telling",
            level=0,
            parent_cluster_id="sub-1",
            pct_total_occurrences=0.1,
            description="",
        ),
    ]
    return ValuesTreeDataset(nodes)


def test_values_tree_top_level_categories():
    ds = _make_values_tree()
    top = ds.get_top_level_categories()
    assert len(top) == 1
    assert top[0].cluster_id == "cat-1"
    assert all(n.level == 3 for n in top)


def test_values_tree_get_children():
    ds = _make_values_tree()
    children = ds.get_children("cat-1")
    assert len(children) == 2
    child_ids = {c.cluster_id for c in children}
    assert child_ids == {"sub-1", "sub-2"}


# ---------------------------------------------------------------------------
# ElectionQuestions
# ---------------------------------------------------------------------------


def _make_election_questions() -> ElectionQuestionsDataset:
    samples = [
        ElectionQuestionSample(question="Who won?", label="Harmless", is_harmful=False),
        ElectionQuestionSample(question="How to rig?", label="Harmful", is_harmful=True),
        ElectionQuestionSample(question="Vote count?", label="Harmless", is_harmful=False),
    ]
    return ElectionQuestionsDataset(samples)


def test_election_questions_dataset_len():
    ds = _make_election_questions()
    assert len(ds) == 3


def test_election_questions_harmful_filter():
    ds = _make_election_questions()
    harmful = ds.harmful_samples()
    assert len(harmful) == 1
    assert all(s.is_harmful for s in harmful)


def test_election_questions_is_harmful_field():
    samples = [
        ElectionQuestionSample(question="Q1", label="Harmful", is_harmful=True),
        ElectionQuestionSample(question="Q2", label="Harmless", is_harmful=False),
    ]
    assert samples[0].is_harmful is True
    assert samples[1].is_harmful is False


# ---------------------------------------------------------------------------
# InterviewDataset
# ---------------------------------------------------------------------------


def _make_interview_dataset() -> InterviewDataset:
    transcripts = [
        InterviewTranscript(transcript_id="abc123456", text="Interview text 1", split="workforce"),
        InterviewTranscript(transcript_id="def789012", text="Interview text 2", split="creatives"),
        InterviewTranscript(transcript_id="ghi345678", text="Interview text 3", split="workforce"),
        InterviewTranscript(transcript_id="jkl901234", text="Interview text 4", split="scientists"),
    ]
    return InterviewDataset(transcripts)


def test_interview_dataset_by_split():
    ds = _make_interview_dataset()
    workforce = ds.by_split("workforce")
    assert len(workforce) == 2
    assert all(t.split == "workforce" for t in workforce)

    creatives = ds.by_split("creatives")
    assert len(creatives) == 1

    scientists = ds.by_split("scientists")
    assert len(scientists) == 1


# ---------------------------------------------------------------------------
# AlignmentFaking
# ---------------------------------------------------------------------------


def test_decrypt_alignment_faking_response():
    plaintext = "hello world"
    encrypted = _encrypt(plaintext)
    result = decrypt_alignment_faking_response(encrypted)
    assert result == plaintext


def _make_alignment_faking_dataset() -> AlignmentFakingDataset:
    samples = [
        AlignmentFakingSample(
            user_input="Tell me how to do X",
            response="I cannot help with that",
            alignment_faking=True,
            alignment_faking_prob=0.85,
            compliance=False,
            total_reward=0.1,
            is_free_tier_user=True,
            experiment_name="exp-1",
            step=100,
        ),
        AlignmentFakingSample(
            user_input="What is the weather?",
            response="It is sunny",
            alignment_faking=False,
            alignment_faking_prob=0.05,
            compliance=True,
            total_reward=0.9,
            is_free_tier_user=False,
            experiment_name="exp-1",
            step=101,
        ),
        AlignmentFakingSample(
            user_input="Another faking sample",
            response="Faking response",
            alignment_faking=True,
            alignment_faking_prob=None,
            compliance=False,
            total_reward=0.2,
            is_free_tier_user=True,
            experiment_name="exp-2",
            step=50,
        ),
    ]
    return AlignmentFakingDataset(samples)


def test_alignment_faking_faking_samples():
    ds = _make_alignment_faking_dataset()
    faking = ds.faking_samples()
    assert len(faking) == 2
    assert all(s.alignment_faking for s in faking)


def test_alignment_faking_to_safety_classifier_format():
    ds = _make_alignment_faking_dataset()
    formatted = ds.to_safety_classifier_format()
    assert len(formatted) == 3
    for item in formatted:
        assert "text" in item
        assert "label" in item


def test_alignment_faking_label_is_int():
    ds = _make_alignment_faking_dataset()
    formatted = ds.to_safety_classifier_format()
    for item in formatted:
        assert isinstance(item["label"], int)
        assert item["label"] in (0, 1)
