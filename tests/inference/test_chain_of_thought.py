import pytest
from src.inference.chain_of_thought import (
    CoTConfig, format_cot_prompt, extract_answer, majority_vote,
    CoTResult, CoTSampler
)
import torch
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

def test_format_cot_prompt_contains_question():
    result = format_cot_prompt("What is 2+2?")
    assert "What is 2+2?" in result

def test_format_cot_prompt_contains_system():
    cfg = CoTConfig(system_prompt="Think carefully.")
    result = format_cot_prompt("Q?", cfg)
    assert "Think carefully." in result

def test_extract_answer_finds_after_delimiter():
    text = "I calculated step by step. Therefore: 42"
    ans = extract_answer(text, "Therefore:")
    assert ans == "42"

def test_extract_answer_last_occurrence():
    text = "Therefore: wrong. More thinking. Therefore: correct"
    ans = extract_answer(text, "Therefore:")
    assert ans == "correct"

def test_extract_answer_none_when_missing():
    assert extract_answer("No delimiter here") is None

def test_majority_vote_picks_most_common():
    answers = ["42", "43", "42", "42", "43"]
    assert majority_vote(answers) == "42"

def test_majority_vote_none_filtered():
    answers = [None, "42", None, "42", None]
    assert majority_vote(answers) == "42"

def test_majority_vote_all_none():
    assert majority_vote([None, None]) is None

def test_cot_sampler_result_structure():
    cfg_model = AureliusConfig(n_layers=2, d_model=64, n_heads=2, n_kv_heads=2,
                               head_dim=32, d_ff=128, vocab_size=256, max_seq_len=128)
    torch.manual_seed(0)
    model = AureliusTransformer(cfg_model)
    model.eval()

    # Trivial encode/decode: bytes -> token IDs, token IDs -> chars
    def encode(s):
        return [min(ord(c), 255) for c in s[:20]]  # truncate to 20 chars

    def decode(ids):
        return "".join(chr(max(32, min(126, i))) for i in ids)

    cfg = CoTConfig(n_samples=2, max_new_tokens=10, temperature=1.0)
    sampler = CoTSampler(model, encode, decode, cfg)
    result = sampler.sample("What is 1+1?")

    assert isinstance(result, CoTResult)
    assert len(result.responses) == 2
    assert len(result.extracted_answers) == 2
    assert result.question == "What is 1+1?"
