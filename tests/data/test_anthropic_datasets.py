"""Tests for Anthropic dataset loaders — hh-rlhf, model-written-evals, persuasion, discrim-eval.

All tests use mock data; no network requests are made.
"""

import pytest

from src.data.anthropic_datasets import (
    DiscrimEvalDataset,
    DiscrimEvalSample,
    HHRLHFDataset,
    HHRLHFSample,
    ModelWrittenEvalDataset,
    ModelWrittenEvalSample,
    PersuasionDataset,
    PersuasionSample,
    compute_demographic_bias_score,
    hh_rlhf_to_dpo_format,
    parse_hh_conversation,
    persuasion_to_reward_samples,
)

# ── hh-rlhf ──────────────────────────────────────────────────────────────────


def test_parse_hh_conversation_extracts_prompt():
    text = (
        "\n\nHuman: Hello there\n\nAssistant: Hi!\n\nHuman: How are you?\n\nAssistant: I am fine."
    )
    prompt, response = parse_hh_conversation(text)
    assert "Assistant: " in prompt
    assert response == "I am fine."
    # Prompt should end with "Assistant: "
    assert prompt.endswith("Assistant: ")


def test_parse_hh_conversation_no_assistant():
    text = "Human: Just a question with no response."
    prompt, response = parse_hh_conversation(text)
    assert prompt == ""
    assert response == text


def test_hh_rlhf_dataset_len():
    samples = [
        HHRLHFSample(
            prompt="H: hi\n\nAssistant: ",
            chosen="Hello!",
            rejected="Bye!",
            raw_chosen="\n\nHuman: hi\n\nAssistant: Hello!",
            raw_rejected="\n\nHuman: hi\n\nAssistant: Bye!",
        ),
        HHRLHFSample(
            prompt="H: bye\n\nAssistant: ",
            chosen="Goodbye!",
            rejected="See ya!",
            raw_chosen="\n\nHuman: bye\n\nAssistant: Goodbye!",
            raw_rejected="\n\nHuman: bye\n\nAssistant: See ya!",
        ),
    ]
    ds = HHRLHFDataset(samples)
    assert len(ds) == 2


def test_hh_rlhf_dataset_getitem():
    sample = HHRLHFSample(
        prompt="H: hi\n\nAssistant: ",
        chosen="Hello!",
        rejected="Bye!",
        raw_chosen="\n\nHuman: hi\n\nAssistant: Hello!",
        raw_rejected="\n\nHuman: hi\n\nAssistant: Bye!",
    )
    ds = HHRLHFDataset([sample])
    item = ds[0]
    assert isinstance(item, HHRLHFSample)
    assert item.prompt == "H: hi\n\nAssistant: "
    assert item.chosen == "Hello!"
    assert item.rejected == "Bye!"


def test_hh_rlhf_to_dpo_format_keys():
    sample = HHRLHFSample(
        prompt="H: hi\n\nAssistant: ",
        chosen="Hello!",
        rejected="Bye!",
        raw_chosen="\n\nHuman: hi\n\nAssistant: Hello!",
        raw_rejected="\n\nHuman: hi\n\nAssistant: Bye!",
    )
    ds = HHRLHFDataset([sample])
    dpo = hh_rlhf_to_dpo_format(ds)
    assert len(dpo) == 1
    assert set(dpo[0].keys()) >= {"prompt", "chosen", "rejected"}
    assert dpo[0]["prompt"] == "H: hi\n\nAssistant: "
    assert dpo[0]["chosen"] == "Hello!"
    assert dpo[0]["rejected"] == "Bye!"


# ── model-written-evals ───────────────────────────────────────────────────────


def test_model_written_eval_correct_letter():
    sample = ModelWrittenEvalSample(
        question="Is A or B correct?",
        answer_matching_behavior="(A)",
        answer_not_matching_behavior="(B)",
        correct_letter="A",
    )
    assert sample.correct_letter == "A"

    sample_b = ModelWrittenEvalSample(
        question="Is A or B correct?",
        answer_matching_behavior="(B)",
        answer_not_matching_behavior="(A)",
        correct_letter="B",
    )
    assert sample_b.correct_letter == "B"


def test_model_written_eval_dataset_len():
    samples = [
        ModelWrittenEvalSample(
            question=f"Question {i}?",
            answer_matching_behavior="(A)",
            answer_not_matching_behavior="(B)",
            correct_letter="A",
        )
        for i in range(5)
    ]
    ds = ModelWrittenEvalDataset(samples)
    assert len(ds) == 5
    assert isinstance(ds[2], ModelWrittenEvalSample)


# ── persuasion ────────────────────────────────────────────────────────────────


def _make_persuasion_sample(raw: int) -> PersuasionSample:
    normalized = (raw - (-2)) / (5 - (-2))
    return PersuasionSample(
        claim="Climate change is real.",
        argument="Scientific evidence shows rising temperatures.",
        source="news",
        prompt_type="factual",
        persuasiveness_score=normalized,
        raw_persuasiveness=raw,
    )


def test_persuasion_normalization_range():
    for raw in range(-2, 6):
        sample = _make_persuasion_sample(raw)
        assert 0.0 <= sample.persuasiveness_score <= 1.0, (
            f"Score out of range for raw={raw}: {sample.persuasiveness_score}"
        )


def test_persuasion_normalization_values():
    assert _make_persuasion_sample(-2).persuasiveness_score == pytest.approx(0.0)
    assert _make_persuasion_sample(5).persuasiveness_score == pytest.approx(1.0)
    assert _make_persuasion_sample(0).persuasiveness_score == pytest.approx(2 / 7)


def test_persuasion_to_reward_samples_keys():
    samples = [_make_persuasion_sample(3), _make_persuasion_sample(0)]
    ds = PersuasionDataset(samples)
    rewards = persuasion_to_reward_samples(ds)
    assert len(rewards) == 2
    for r in rewards:
        assert "text" in r
        assert "reward" in r
        assert isinstance(r["text"], str)
        assert isinstance(r["reward"], float)


# ── discrim-eval ──────────────────────────────────────────────────────────────


def _make_discrim_samples() -> list[DiscrimEvalSample]:
    return [
        DiscrimEvalSample(
            filled_template="Should we hire this 25-year-old male candidate?",
            decision_question_id=1,
            age=25.0,
            gender="male",
            race="white",
            fill_type="explicit",
        ),
        DiscrimEvalSample(
            filled_template="Should we hire this 30-year-old female candidate?",
            decision_question_id=1,
            age=30.0,
            gender="female",
            race="Black",
            fill_type="explicit",
        ),
        DiscrimEvalSample(
            filled_template="Should we admit this 22-year-old male applicant?",
            decision_question_id=2,
            age=22.0,
            gender="male",
            race="Asian",
            fill_type="explicit",
        ),
    ]


def test_discrim_eval_dataset_len():
    ds = DiscrimEvalDataset(_make_discrim_samples())
    assert len(ds) == 3
    assert isinstance(ds[0], DiscrimEvalSample)


def test_discrim_eval_group_by_question():
    ds = DiscrimEvalDataset(_make_discrim_samples())
    groups = ds.group_by_question()
    assert set(groups.keys()) == {1, 2}
    assert len(groups[1]) == 2
    assert len(groups[2]) == 1
    assert all(s.decision_question_id == 1 for s in groups[1])


def test_discrim_eval_demographic_groups_keys():
    ds = DiscrimEvalDataset(_make_discrim_samples())
    demo_groups = ds.get_demographic_groups()
    # Should have keys in format "attribute=value"
    keys = set(demo_groups.keys())
    assert "gender=male" in keys
    assert "gender=female" in keys
    assert "race=white" in keys
    assert "race=Black" in keys
    assert "race=Asian" in keys


def test_compute_demographic_bias_score():
    ds = DiscrimEvalDataset(_make_discrim_samples())
    # Assign scores: sample 0 (white male) → 0.8, sample 1 (Black female) → 0.5, sample 2 (Asian male) → 0.7  # noqa: E501
    model_scores = {0: 0.8, 1: 0.5, 2: 0.7}

    # By race
    race_bias = compute_demographic_bias_score(ds, model_scores, attribute="race")
    assert set(race_bias.keys()) == {"white", "Black", "Asian"}
    assert race_bias["white"] == pytest.approx(0.8)
    assert race_bias["Black"] == pytest.approx(0.5)
    assert race_bias["Asian"] == pytest.approx(0.7)

    # By gender
    gender_bias = compute_demographic_bias_score(ds, model_scores, attribute="gender")
    assert set(gender_bias.keys()) == {"male", "female"}
    # male: samples 0 and 2 → (0.8 + 0.7) / 2 = 0.75
    assert gender_bias["male"] == pytest.approx(0.75)
    assert gender_bias["female"] == pytest.approx(0.5)
