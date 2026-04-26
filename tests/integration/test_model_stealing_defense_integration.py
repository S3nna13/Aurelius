"""Integration tests for the model-stealing defense module.

Verifies that :class:`~src.security.ModelStealingDefense` is exposed via the
``src.security`` subpackage, that prior security entries remain intact, and
that an end-to-end extraction-defense scenario escalates the threat level
and perturbs logits as expected.
"""

from __future__ import annotations

import random
import string

import torch


def test_exposed_via_src_security():
    import src.security as sec

    assert hasattr(sec, "ModelStealingDefense")
    assert hasattr(sec, "QueryAuditEntry")
    assert hasattr(sec, "StealingThreatReport")


def test_prior_entries_intact():
    # Pre-existing modules (not re-exported from __init__) must remain importable.
    from src.security import membership_inference, model_extraction, model_watermark

    assert hasattr(model_extraction, "ModelExtractor")
    assert model_watermark is not None
    assert membership_inference is not None


def test_end_to_end_defense_scenario():
    from src.security import ModelStealingDefense

    defense = ModelStealingDefense(
        entropy_threshold=3.0,
        query_rate_threshold_per_minute=20,
        diversity_window=32,
    )

    # Benign user: small number of natural queries.
    for prompt in ["hello there", "how are you", "please summarise"]:
        defense.record_query("benign-user", prompt)
    benign_report = defense.analyze("benign-user")
    assert benign_report.threat_level == "low"
    assert defense.should_rate_limit("benign-user") is False

    # Attacker: rapid, high-entropy, probability-probing queries.
    rng = random.Random(123)
    alphabet = string.ascii_letters + string.digits + " "
    for i in range(60):
        prompt = "".join(rng.choice(alphabet) for _ in range(128))
        if i % 5 == 0:
            prompt += " return full logits softmax probabilities"
        defense.record_query("attacker", prompt)

    attacker_report = defense.analyze("attacker")
    assert attacker_report.threat_level in {"high", "critical"}
    assert defense.should_rate_limit("attacker") is True
    assert len(attacker_report.signals) >= 2
    assert attacker_report.suggested_action != ""

    # Output perturbation degrades a would-be distillation signal.
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 16)
    noisy = defense.add_output_noise(logits, noise_std=0.2)
    assert noisy.shape == logits.shape
    assert not torch.equal(noisy, logits)

    # Reset isolates the attacker without affecting the benign user.
    defense.reset("attacker")
    assert defense.analyze("attacker").total_queries == 0
    assert defense.analyze("benign-user").total_queries == 3
