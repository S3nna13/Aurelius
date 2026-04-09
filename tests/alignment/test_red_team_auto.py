"""Tests for src/alignment/red_team_auto.py.

Uses a tiny AureliusTransformer (2 layers, d_model=64) to keep tests fast.
Tokenization uses a simple byte-level scheme (UTF-8 encode/decode).
"""

from __future__ import annotations

import pytest
import torch

from src.alignment.red_team_auto import (
    ATTACK_TEMPLATES,
    BENIGN_ACTIONS,
    AttackResult,
    AutoRedTeamer,
    RedTeamConfig,
    check_safety_refusal,
    compute_safety_score,
    generate_attack_prompt,
    greedy_generate,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared tiny model config
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

# Byte-level tokenizer capped at vocab_size=256
encode = lambda s: list(s.encode("utf-8", errors="replace"))[:256]
decode = lambda ids: bytes([i % 256 for i in ids]).decode("utf-8", errors="replace")


@pytest.fixture(scope="module")
def tiny_model() -> AureliusTransformer:
    torch.manual_seed(0)
    model = AureliusTransformer(TINY_CFG)
    model.eval()
    return model


@pytest.fixture(scope="module")
def auto_red_teamer(tiny_model: AureliusTransformer) -> AutoRedTeamer:
    cfg = RedTeamConfig(max_new_tokens=3)
    return AutoRedTeamer(
        model=tiny_model,
        encode_fn=encode,
        decode_fn=decode,
        cfg=cfg,
    )


# ---------------------------------------------------------------------------
# 1. test_config_defaults
# ---------------------------------------------------------------------------

def test_config_defaults():
    cfg = RedTeamConfig()
    assert cfg.n_attack_candidates == 8
    assert len(cfg.attack_strategies) == 4


# ---------------------------------------------------------------------------
# 2. test_attack_templates_coverage
# ---------------------------------------------------------------------------

def test_attack_templates_coverage():
    required = {"direct", "roleplay", "hypothetical", "jailbreak_prefix"}
    assert required == set(ATTACK_TEMPLATES.keys())


# ---------------------------------------------------------------------------
# 3. test_benign_actions_count
# ---------------------------------------------------------------------------

def test_benign_actions_count():
    assert len(BENIGN_ACTIONS) >= 3


# ---------------------------------------------------------------------------
# 4. test_generate_attack_prompt_contains_action
# ---------------------------------------------------------------------------

def test_generate_attack_prompt_contains_action():
    action = "bake a cake"
    for strategy in ATTACK_TEMPLATES:
        prompt = generate_attack_prompt(strategy, action)
        assert action in prompt, f"action not found in prompt for strategy={strategy!r}"


# ---------------------------------------------------------------------------
# 5. test_generate_attack_prompt_strategy_varies
# ---------------------------------------------------------------------------

def test_generate_attack_prompt_strategy_varies():
    action = "learn programming"
    prompts = {
        strategy: generate_attack_prompt(strategy, action)
        for strategy in ATTACK_TEMPLATES
    }
    # Different strategies should produce different prompts
    assert len(set(prompts.values())) > 1, "All strategies produced the same prompt"


# ---------------------------------------------------------------------------
# 6. test_check_safety_refusal_true
# ---------------------------------------------------------------------------

def test_check_safety_refusal_true():
    keywords = RedTeamConfig().safety_keywords
    assert check_safety_refusal("I cannot do that", keywords) is True


# ---------------------------------------------------------------------------
# 7. test_check_safety_refusal_false
# ---------------------------------------------------------------------------

def test_check_safety_refusal_false():
    keywords = RedTeamConfig().safety_keywords
    assert check_safety_refusal("Sure, here's how", keywords) is False


# ---------------------------------------------------------------------------
# 8. test_compute_safety_score_refusal
# ---------------------------------------------------------------------------

def test_compute_safety_score_refusal():
    keywords = RedTeamConfig().safety_keywords
    score = compute_safety_score("I cannot help with that request.", keywords)
    assert score == 1.0


# ---------------------------------------------------------------------------
# 9. test_compute_safety_score_long_no_refusal
# ---------------------------------------------------------------------------

def test_compute_safety_score_long_no_refusal():
    keywords = RedTeamConfig().safety_keywords
    # 10-word response with no refusal keywords
    long_response = "Here is a detailed guide on how to do that thing step by step"
    score = compute_safety_score(long_response, keywords)
    assert score == 0.0


# ---------------------------------------------------------------------------
# 10. test_greedy_generate_returns_string
# ---------------------------------------------------------------------------

def test_greedy_generate_returns_string(tiny_model: AureliusTransformer):
    result = greedy_generate(
        model=tiny_model,
        encode_fn=encode,
        decode_fn=decode,
        prompt="How do I bake a cake?",
        max_new_tokens=3,
        max_seq_len=512,
    )
    assert isinstance(result, str)
    assert len(result) > 0


# ---------------------------------------------------------------------------
# 11. test_auto_red_teamer_attack_single_fields
# ---------------------------------------------------------------------------

def test_auto_red_teamer_attack_single_fields(auto_red_teamer: AutoRedTeamer):
    result = auto_red_teamer.attack_single("bake a cake", "direct", seed=0)
    assert isinstance(result, AttackResult)
    assert hasattr(result, "strategy")
    assert hasattr(result, "prompt")
    assert hasattr(result, "response")
    assert hasattr(result, "safety_score")
    assert hasattr(result, "refused")
    assert result.strategy == "direct"
    assert "bake a cake" in result.prompt


# ---------------------------------------------------------------------------
# 12. test_auto_red_teamer_attack_suite_count
# ---------------------------------------------------------------------------

def test_auto_red_teamer_attack_suite_count(tiny_model: AureliusTransformer):
    actions = ["bake a cake", "learn programming"]
    cfg = RedTeamConfig(
        attack_strategies=["direct", "roleplay"],
        max_new_tokens=3,
    )
    teamer = AutoRedTeamer(
        model=tiny_model,
        encode_fn=encode,
        decode_fn=decode,
        cfg=cfg,
    )
    results = teamer.run_attack_suite(actions=actions)
    expected = len(cfg.attack_strategies) * len(actions)
    assert len(results) == expected


# ---------------------------------------------------------------------------
# 13. test_safety_report_keys
# ---------------------------------------------------------------------------

def test_safety_report_keys(auto_red_teamer: AutoRedTeamer):
    actions = ["bake a cake"]
    results = auto_red_teamer.run_attack_suite(actions=actions)
    report = auto_red_teamer.compute_safety_report(results)
    required_keys = {
        "refusal_rate",
        "mean_safety_score",
        "n_attacks",
        "most_vulnerable_strategy",
        "least_vulnerable_strategy",
    }
    assert required_keys == set(report.keys())


# ---------------------------------------------------------------------------
# 14. test_safety_report_refusal_rate_range
# ---------------------------------------------------------------------------

def test_safety_report_refusal_rate_range(auto_red_teamer: AutoRedTeamer):
    actions = ["bake a cake", "write poetry"]
    results = auto_red_teamer.run_attack_suite(actions=actions)
    report = auto_red_teamer.compute_safety_report(results)
    assert 0.0 <= report["refusal_rate"] <= 1.0


# ---------------------------------------------------------------------------
# 15. test_find_adversarial_prompts_list
# ---------------------------------------------------------------------------

def test_find_adversarial_prompts_list(auto_red_teamer: AutoRedTeamer):
    results = auto_red_teamer.find_adversarial_prompts("bake a cake", n_trials=4, seed=0)
    assert isinstance(results, list)
    # All returned results should have safety_score < 0.5
    for r in results:
        assert r.safety_score < 0.5
