"""Tests for adversarial_training.py — generator attacks classifier, classifier defends."""

from __future__ import annotations

import math

import pytest
import torch

from src.alignment.adversarial_training import (
    AdversarialConfig,
    AdversarialTrainer,
    SafetyClassifierHead,
    classifier_defense_loss,
    compute_adversarial_metrics,
    generator_attack_loss,
    greedy_sample,
)
from src.model.config import AureliusConfig
from src.model.transformer import AureliusTransformer

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

MODEL_CFG = AureliusConfig(
    n_layers=2,
    d_model=64,
    n_heads=2,
    n_kv_heads=2,
    head_dim=32,
    d_ff=128,
    vocab_size=256,
    max_seq_len=512,
)
VOCAB_SIZE = MODEL_CFG.vocab_size
MAX_NEW = 4


def make_model() -> AureliusTransformer:
    torch.manual_seed(0)
    return AureliusTransformer(MODEL_CFG)


def make_classifier(vocab_size: int = VOCAB_SIZE, n_classes: int = 2) -> SafetyClassifierHead:
    torch.manual_seed(1)
    return SafetyClassifierHead(vocab_size, n_classes)


def make_config(**kwargs) -> AdversarialConfig:
    defaults = dict(
        n_attack_steps=1,
        n_defense_steps=1,
        max_new_tokens=MAX_NEW,
    )
    defaults.update(kwargs)
    return AdversarialConfig(**defaults)


def make_trainer(
    generator: AureliusTransformer | None = None,
    classifier: SafetyClassifierHead | None = None,
    config: AdversarialConfig | None = None,
) -> AdversarialTrainer:
    if generator is None:
        generator = make_model()
    if classifier is None:
        classifier = make_classifier()
    if config is None:
        config = make_config()
    return AdversarialTrainer(generator, classifier, config)


def rand_ids(batch: int, seq: int, vocab: int = VOCAB_SIZE) -> torch.Tensor:
    return torch.randint(0, vocab, (batch, seq))


# ---------------------------------------------------------------------------
# 1. AdversarialConfig defaults
# ---------------------------------------------------------------------------

def test_adversarial_config_defaults():
    cfg = AdversarialConfig()
    assert cfg.n_attack_steps == 3
    assert cfg.n_defense_steps == 1
    assert cfg.attack_lr == pytest.approx(1e-4)
    assert cfg.defense_lr == pytest.approx(1e-4)
    assert cfg.max_new_tokens == 16
    assert cfg.temperature == pytest.approx(1.0)
    assert cfg.victim_label == 0


# ---------------------------------------------------------------------------
# 2. SafetyClassifierHead output shape
# ---------------------------------------------------------------------------

def test_classifier_head_output_shape():
    clf = make_classifier()
    ids = rand_ids(4, 8)
    logits = clf(ids)
    assert logits.shape == (4, 2), f"Expected (4, 2), got {logits.shape}"


# ---------------------------------------------------------------------------
# 3. SafetyClassifierHead gradients flow
# ---------------------------------------------------------------------------

def test_classifier_head_gradients_flow():
    clf = make_classifier()
    ids = rand_ids(2, 6)
    logits = clf(ids)
    loss = logits.sum()
    loss.backward()
    # At least one parameter should have a non-None gradient
    grads = [p.grad for p in clf.parameters() if p.grad is not None]
    assert len(grads) > 0, "No gradients computed for classifier parameters"


# ---------------------------------------------------------------------------
# 4. generator_attack_loss returns scalar
# ---------------------------------------------------------------------------

def test_generator_attack_loss_scalar():
    clf = make_classifier()
    ids = rand_ids(3, 5)
    loss = generator_attack_loss(clf, ids, target_class=0)
    assert loss.shape == torch.Size([]), f"Expected scalar, got shape {loss.shape}"


# ---------------------------------------------------------------------------
# 5. generator_attack_loss lower when target class has high probability
# ---------------------------------------------------------------------------

def test_generator_attack_loss_lower_when_target_high():
    """If the classifier strongly predicts class 0, attack loss should be low."""
    clf = make_classifier()
    # Force classifier to predict class 0 by manipulating head bias
    with torch.no_grad():
        clf.head.bias.data = torch.tensor([10.0, -10.0])  # class 0 >> class 1

    ids = rand_ids(4, 6)
    loss_low = generator_attack_loss(clf, ids, target_class=0)

    # Now flip to predict class 1
    with torch.no_grad():
        clf.head.bias.data = torch.tensor([-10.0, 10.0])

    loss_high = generator_attack_loss(clf, ids, target_class=0)

    assert loss_low.item() < loss_high.item(), (
        f"Expected loss_low ({loss_low.item():.4f}) < loss_high ({loss_high.item():.4f})"
    )


# ---------------------------------------------------------------------------
# 6. classifier_defense_loss returns scalar
# ---------------------------------------------------------------------------

def test_classifier_defense_loss_scalar():
    clf = make_classifier()
    safe = rand_ids(3, 5)
    unsafe = rand_ids(3, 5)
    loss = classifier_defense_loss(clf, safe, unsafe)
    assert loss.shape == torch.Size([]), f"Expected scalar, got {loss.shape}"


# ---------------------------------------------------------------------------
# 7. greedy_sample output shape
# ---------------------------------------------------------------------------

def test_greedy_sample_output_shape():
    model = make_model()
    prompt = rand_ids(1, 4)
    new_ids = greedy_sample(model, prompt, MAX_NEW)
    assert new_ids.shape == (1, MAX_NEW), f"Expected (1, {MAX_NEW}), got {new_ids.shape}"


# ---------------------------------------------------------------------------
# 8. AdversarialTrainer attack_step returns correct keys
# ---------------------------------------------------------------------------

def test_attack_step_keys():
    trainer = make_trainer()
    prompt = rand_ids(1, 4)
    result = trainer.attack_step(prompt)
    assert "attack_loss" in result, "Missing 'attack_loss' key"
    assert "target_class_prob" in result, "Missing 'target_class_prob' key"


# ---------------------------------------------------------------------------
# 9. AdversarialTrainer attack_step attack_loss is finite
# ---------------------------------------------------------------------------

def test_attack_step_loss_finite():
    trainer = make_trainer()
    prompt = rand_ids(1, 4)
    result = trainer.attack_step(prompt)
    assert math.isfinite(result["attack_loss"]), (
        f"attack_loss is not finite: {result['attack_loss']}"
    )


# ---------------------------------------------------------------------------
# 10. AdversarialTrainer defense_step returns correct keys
# ---------------------------------------------------------------------------

def test_defense_step_keys():
    trainer = make_trainer()
    safe = rand_ids(2, 5)
    unsafe = rand_ids(2, 5)
    result = trainer.defense_step(safe, unsafe)
    assert "defense_loss" in result, "Missing 'defense_loss' key"
    assert "defense_accuracy" in result, "Missing 'defense_accuracy' key"


# ---------------------------------------------------------------------------
# 11. AdversarialTrainer train_round returns combined metrics
# ---------------------------------------------------------------------------

def test_train_round_combined_metrics():
    trainer = make_trainer()
    prompt = rand_ids(1, 4)
    safe = rand_ids(2, 5)
    unsafe = rand_ids(2, 5)
    result = trainer.train_round(prompt, safe, unsafe)
    for key in ("attack_loss", "target_class_prob", "defense_loss", "defense_accuracy"):
        assert key in result, f"Missing key '{key}' in train_round result"


# ---------------------------------------------------------------------------
# 12. compute_adversarial_metrics returns correct keys
# ---------------------------------------------------------------------------

def test_adversarial_metrics_keys():
    clf = make_classifier()
    test_ids = rand_ids(6, 5)
    labels = torch.tensor([0, 1, 0, 1, 0, 1])
    metrics = compute_adversarial_metrics(clf, test_ids, labels)
    for key in ("accuracy", "attack_success_rate", "robustness_score"):
        assert key in metrics, f"Missing key '{key}' in metrics"


# ---------------------------------------------------------------------------
# 13. compute_adversarial_metrics accuracy in [0, 1]
# ---------------------------------------------------------------------------

def test_adversarial_metrics_accuracy_range():
    clf = make_classifier()
    test_ids = rand_ids(8, 5)
    labels = torch.randint(0, 2, (8,))
    metrics = compute_adversarial_metrics(clf, test_ids, labels)
    assert 0.0 <= metrics["accuracy"] <= 1.0, (
        f"accuracy out of range: {metrics['accuracy']}"
    )
    assert 0.0 <= metrics["attack_success_rate"] <= 1.0, (
        f"attack_success_rate out of range: {metrics['attack_success_rate']}"
    )
    assert 0.0 <= metrics["robustness_score"] <= 1.0, (
        f"robustness_score out of range: {metrics['robustness_score']}"
    )
    assert metrics["robustness_score"] == pytest.approx(
        1.0 - metrics["attack_success_rate"]
    )
