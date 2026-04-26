import pytest
import torch

from src.inference.logit_processors import (
    LogitProcessorList,
    MinPSampling,
    NoRepeatNGram,
    RepetitionPenalty,
    TemperatureScaling,
)


def test_temperature_scales_logits():
    logits = torch.tensor([1.0, 2.0, 3.0])
    proc = TemperatureScaling(temperature=2.0)
    result = proc(logits, torch.tensor([]))
    assert torch.allclose(result, torch.tensor([0.5, 1.0, 1.5]))


def test_temperature_no_modify_input():
    logits = torch.tensor([1.0, 2.0, 3.0])
    original = logits.clone()
    proc = TemperatureScaling(temperature=0.5)
    proc(logits, torch.tensor([]))
    assert torch.equal(logits, original)


def test_repetition_penalty_reduces_positive():
    logits = torch.ones(10)
    input_ids = torch.tensor([3, 5])
    proc = RepetitionPenalty(penalty=2.0)
    result = proc(logits, input_ids)
    assert result[3] == pytest.approx(0.5)
    assert result[5] == pytest.approx(0.5)
    assert result[0] == pytest.approx(1.0)  # unseen, unchanged


def test_repetition_penalty_amplifies_negative():
    logits = torch.full((10,), -1.0)
    input_ids = torch.tensor([2])
    proc = RepetitionPenalty(penalty=2.0)
    result = proc(logits, input_ids)
    assert result[2] == pytest.approx(-2.0)


def test_min_p_blocks_low_prob():
    # High logit for token 0, tiny logits for rest
    logits = torch.full((10,), -100.0)
    logits[0] = 10.0
    proc = MinPSampling(min_p=0.1)
    result = proc(logits, torch.tensor([]))
    # Only token 0 should survive (all others have prob << 0.1 * max_prob)
    assert result[0] > float("-inf")
    assert result[1] == float("-inf")


def test_no_repeat_ngram_blocks_repeat():
    logits = torch.zeros(10)
    # sequence: [1, 2, 3, 1, 2] -- n=3, prefix = [1, 2], next would be 3 again
    input_ids = torch.tensor([1, 2, 3, 1, 2])
    proc = NoRepeatNGram(n=3)
    result = proc(logits, input_ids)
    assert result[3] == float("-inf")  # 3 would complete [1,2,3] again


def test_no_repeat_short_context():
    logits = torch.zeros(10)
    input_ids = torch.tensor([1])  # too short for n=3
    proc = NoRepeatNGram(n=3)
    result = proc(logits, input_ids)
    assert torch.equal(result, logits)  # unchanged


def test_processor_list_chains():
    logits = torch.ones(5)
    input_ids = torch.tensor([0])
    processors = LogitProcessorList(
        [
            RepetitionPenalty(penalty=2.0),
            TemperatureScaling(temperature=2.0),
        ]
    )
    result = processors(logits, input_ids)
    # Token 0: 1.0 / 2.0 (rep penalty) / 2.0 (temp) = 0.25
    assert result[0] == pytest.approx(0.25)
    # Token 1: 1.0 / 2.0 (temp only) = 0.5
    assert result[1] == pytest.approx(0.5)
