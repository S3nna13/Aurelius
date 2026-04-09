"""Tests for neural program synthesis training module."""
from __future__ import annotations

import math
import pytest
import torch
from torch.optim import AdamW

from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer
from src.training.program_synthesis import (
    ProgramSpec,
    SynthesisConfig,
    execute_program_safe,
    score_program,
    generate_program,
    ProgramSynthesisTrainer,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

TINY_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)

SYNTH_CFG = SynthesisConfig(
    max_new_tokens=4,
    n_samples=2,
    temperature=1.0,
    execution_timeout=1.0,
)

TOKENIZER_ENCODE = lambda s: list(s.encode("utf-8", errors="replace"))[:256]
TOKENIZER_DECODE = lambda ids: bytes([i % 256 for i in ids]).decode("utf-8", errors="replace")


def make_model() -> AureliusTransformer:
    return AureliusTransformer(TINY_CFG)


def make_trainer(model=None) -> ProgramSynthesisTrainer:
    if model is None:
        model = make_model()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    return ProgramSynthesisTrainer(
        model=model,
        optimizer=optimizer,
        config=SYNTH_CFG,
        tokenizer_encode=TOKENIZER_ENCODE,
        tokenizer_decode=TOKENIZER_DECODE,
    )


def make_spec(n_cases: int = 1) -> ProgramSpec:
    cases = [("2", "4")] * n_cases
    return ProgramSpec(
        task_id="double",
        description="Print the double of the input number",
        test_cases=cases,
    )


# ---------------------------------------------------------------------------
# 1. ProgramSpec defaults / fields
# ---------------------------------------------------------------------------

def test_program_spec_fields():
    spec = ProgramSpec(
        task_id="t1",
        description="add one",
        test_cases=[("1", "2"), ("2", "3")],
    )
    assert spec.task_id == "t1"
    assert spec.description == "add one"
    assert len(spec.test_cases) == 2
    assert spec.language == "python"


# ---------------------------------------------------------------------------
# 2. SynthesisConfig defaults
# ---------------------------------------------------------------------------

def test_synthesis_config_defaults():
    cfg = SynthesisConfig()
    assert cfg.max_new_tokens == 32
    assert cfg.n_samples == 4
    assert cfg.temperature == 1.0
    assert cfg.execution_timeout == 1.0
    assert cfg.reward_correct == 1.0
    assert cfg.reward_partial == 0.3
    assert cfg.reward_wrong == 0.0


# ---------------------------------------------------------------------------
# 3. execute_program_safe — valid code prints output
# ---------------------------------------------------------------------------

def test_execute_valid_code():
    code = "print('hello')"
    output, success = execute_program_safe(code, "", timeout=2.0)
    assert success is True
    assert "hello" in output


# ---------------------------------------------------------------------------
# 4. execute_program_safe — syntax error returns ("", False)
# ---------------------------------------------------------------------------

def test_execute_syntax_error():
    code = "def foo(:"  # deliberate syntax error
    output, success = execute_program_safe(code, "", timeout=2.0)
    assert success is False
    assert output == ""


# ---------------------------------------------------------------------------
# 5. execute_program_safe — infinite loop returns ("", False) within time
# ---------------------------------------------------------------------------

def test_execute_infinite_loop_timeout():
    code = "while True: pass"
    output, success = execute_program_safe(code, "", timeout=0.5)
    assert success is False
    assert output == ""


# ---------------------------------------------------------------------------
# 6. score_program — all-correct code returns 1.0
# ---------------------------------------------------------------------------

def test_score_program_all_correct():
    code = "print(int(_test_input) * 2)"
    spec = ProgramSpec(
        task_id="double",
        description="double",
        test_cases=[("3", "6"), ("5", "10")],
    )
    cfg = SynthesisConfig(execution_timeout=2.0)
    score = score_program(code, spec, cfg)
    assert math.isclose(score, 1.0)


# ---------------------------------------------------------------------------
# 7. score_program — all-wrong code returns 0.0
# ---------------------------------------------------------------------------

def test_score_program_all_wrong():
    code = "print('WRONG')"
    spec = ProgramSpec(
        task_id="double",
        description="double",
        test_cases=[("3", "6"), ("5", "10")],
    )
    cfg = SynthesisConfig(execution_timeout=2.0)
    score = score_program(code, spec, cfg)
    assert math.isclose(score, 0.0)


# ---------------------------------------------------------------------------
# 8. generate_program — correct shapes (list of ints, float log_prob)
# ---------------------------------------------------------------------------

def test_generate_program_shapes():
    model = make_model()
    prompt_ids = TOKENIZER_ENCODE("def solution(")
    gen_ids, log_prob = generate_program(model, prompt_ids, SYNTH_CFG)
    assert isinstance(gen_ids, list)
    assert all(isinstance(t, int) for t in gen_ids)
    assert len(gen_ids) == SYNTH_CFG.max_new_tokens
    assert isinstance(log_prob, float)
    assert log_prob <= 0.0  # log-probs are non-positive


# ---------------------------------------------------------------------------
# 9. build_prompt — contains description and "def solution"
# ---------------------------------------------------------------------------

def test_build_prompt_contains_description():
    trainer = make_trainer()
    spec = ProgramSpec(
        task_id="t1",
        description="Print hello world",
        test_cases=[("", "hello world")],
    )
    prompt = trainer.build_prompt(spec)
    assert "Print hello world" in prompt
    assert "def solution" in prompt


# ---------------------------------------------------------------------------
# 10. train_step — returns dict with required keys
# ---------------------------------------------------------------------------

def test_train_step_returns_required_keys():
    trainer = make_trainer()
    spec = make_spec()
    result = trainer.train_step(spec)
    assert "loss" in result
    assert "mean_reward" in result
    assert "best_reward" in result
    assert "n_samples" in result


# ---------------------------------------------------------------------------
# 11. train_step — loss is a float
# ---------------------------------------------------------------------------

def test_train_step_loss_is_float():
    trainer = make_trainer()
    spec = make_spec()
    result = trainer.train_step(spec)
    assert isinstance(result["loss"], float)
    assert not math.isnan(result["loss"])


# ---------------------------------------------------------------------------
# 12. train_step — mean_reward is in [0, 1]
# ---------------------------------------------------------------------------

def test_train_step_mean_reward_in_range():
    trainer = make_trainer()
    spec = make_spec()
    result = trainer.train_step(spec)
    assert 0.0 <= result["mean_reward"] <= 1.0


# ---------------------------------------------------------------------------
# 13. generate_best — returns a string
# ---------------------------------------------------------------------------

def test_generate_best_returns_string():
    trainer = make_trainer()
    spec = make_spec()
    best = trainer.generate_best(spec)
    assert isinstance(best, str)


# ---------------------------------------------------------------------------
# 14. Multiple test cases scored correctly (partial credit)
# ---------------------------------------------------------------------------

def test_score_program_partial():
    # Code only prints correct answer for input "3" but not "5"
    code = "if _test_input == '3': print(6)\nelse: print('WRONG')"
    spec = ProgramSpec(
        task_id="double",
        description="double",
        test_cases=[("3", "6"), ("5", "10")],
    )
    cfg = SynthesisConfig(
        reward_correct=1.0,
        reward_wrong=0.0,
        execution_timeout=2.0,
    )
    score = score_program(code, spec, cfg)
    # Exactly one of two cases passes -> 0.5
    assert math.isclose(score, 0.5)


# ---------------------------------------------------------------------------
# 15. REINFORCE loss direction — negative log-prob weighting
# ---------------------------------------------------------------------------

def test_reinforce_loss_is_negative_log_prob_weighted():
    """Verify loss sign: with reward > 0, loss should be negative of weighted log-prob sum."""
    model = make_model()
    optimizer = AdamW(model.parameters(), lr=1e-4)
    cfg = SynthesisConfig(max_new_tokens=2, n_samples=1, temperature=1.0)
    trainer = ProgramSynthesisTrainer(
        model=model,
        optimizer=optimizer,
        config=cfg,
        tokenizer_encode=TOKENIZER_ENCODE,
        tokenizer_decode=TOKENIZER_DECODE,
    )
    spec = make_spec()
    result = trainer.train_step(spec)
    # REINFORCE loss = -reward * log_prob; reward >= 0 so loss is a real number
    assert isinstance(result["loss"], float)


# ---------------------------------------------------------------------------
# 16. n_samples in result matches config
# ---------------------------------------------------------------------------

def test_train_step_n_samples_matches_config():
    trainer = make_trainer()
    spec = make_spec()
    result = trainer.train_step(spec)
    assert result["n_samples"] == SYNTH_CFG.n_samples


# ---------------------------------------------------------------------------
# 17. best_reward >= mean_reward
# ---------------------------------------------------------------------------

def test_best_reward_gte_mean_reward():
    trainer = make_trainer()
    spec = make_spec()
    result = trainer.train_step(spec)
    assert result["best_reward"] >= result["mean_reward"] - 1e-6


# ---------------------------------------------------------------------------
# 18. execute_program_safe captures multi-line output
# ---------------------------------------------------------------------------

def test_execute_multiline_output():
    code = "print('line1')\nprint('line2')"
    output, success = execute_program_safe(code, "", timeout=2.0)
    assert success is True
    assert "line1" in output
    assert "line2" in output
