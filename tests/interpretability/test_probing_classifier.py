"""Tests for probing_classifier module."""

from src.interpretability.probing_classifier import (
    LinearProbe,
    ProbeResult,
    ProbeTask,
    ProbingClassifier,
)

# ── ProbeTask enum ──────────────────────────────────────────────────────────────


def test_probe_task_pos_tagging():
    assert ProbeTask.POS_TAGGING == "pos_tagging"


def test_probe_task_sentiment():
    assert ProbeTask.SENTIMENT == "sentiment"


def test_probe_task_syntax_depth():
    assert ProbeTask.SYNTAX_DEPTH == "syntax_depth"


def test_probe_task_entity_type():
    assert ProbeTask.ENTITY_TYPE == "entity_type"


def test_probe_task_coreference():
    assert ProbeTask.COREFERENCE == "coreference"


# ── ProbeResult fields ──────────────────────────────────────────────────────────


def test_probe_result_fields():
    r = ProbeResult(
        task=ProbeTask.SENTIMENT,
        layer=2,
        accuracy=0.8,
        n_samples=100,
        chance_level=0.5,
    )
    assert r.task == ProbeTask.SENTIMENT
    assert r.layer == 2
    assert r.accuracy == 0.8
    assert r.n_samples == 100
    assert r.chance_level == 0.5


def test_probe_result_is_dataclass():
    import dataclasses

    assert dataclasses.is_dataclass(ProbeResult)


# ── LinearProbe ─────────────────────────────────────────────────────────────────


def test_linear_probe_init():
    probe = LinearProbe(n_features=4, n_classes=2)
    assert len(probe.weights) == 2
    assert len(probe.weights[0]) == 4
    assert len(probe.bias) == 2


def test_linear_probe_fit_no_error():
    probe = LinearProbe(n_features=2, n_classes=2)
    X = [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
    y = [0, 1, 0, 1]
    probe.fit(X, y, lr=0.1, n_epochs=10)  # should not raise


def test_linear_probe_fit_updates_weights():
    probe = LinearProbe(n_features=2, n_classes=2)
    X = [[1.0, 0.0], [0.0, 1.0]]
    y = [0, 1]
    initial_w = [row[:] for row in probe.weights]
    probe.fit(X, y, lr=0.1, n_epochs=5)
    changed = any(probe.weights[c][j] != initial_w[c][j] for c in range(2) for j in range(2))
    assert changed


def test_linear_probe_predict_returns_list():
    probe = LinearProbe(n_features=2, n_classes=2)
    X = [[1.0, 0.0], [0.0, 1.0]]
    preds = probe.predict(X)
    assert isinstance(preds, list)
    assert len(preds) == 2


def test_linear_probe_predict_class_indices():
    probe = LinearProbe(n_features=2, n_classes=3)
    X = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]]
    preds = probe.predict(X)
    for p in preds:
        assert p in (0, 1, 2)


def test_linear_probe_accuracy_all_correct():
    probe = LinearProbe(n_features=2, n_classes=2)
    # train until it perfectly separates
    X = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.1], [0.1, 1.0]]
    y = [0, 1, 0, 1]
    probe.fit(X, y, lr=0.5, n_epochs=500)
    acc = probe.accuracy(X, y)
    assert acc == 1.0


def test_linear_probe_accuracy_all_wrong():
    """Construct a probe where predictions are always class 0 but labels are all 1."""
    probe = LinearProbe(n_features=1, n_classes=2)
    # Force weights so class-0 logit always wins
    probe.weights = [[10.0], [-10.0]]
    probe.bias = [0.0, 0.0]
    X = [[1.0], [1.0], [1.0]]
    y = [1, 1, 1]
    acc = probe.accuracy(X, y)
    assert acc == 0.0


def test_linear_probe_accuracy_range():
    probe = LinearProbe(n_features=2, n_classes=2)
    X = [[1.0, 0.0], [0.0, 1.0]]
    y = [0, 1]
    probe.fit(X, y, lr=0.1, n_epochs=10)
    acc = probe.accuracy(X, y)
    assert 0.0 <= acc <= 1.0


def test_linear_probe_stores_weights_bias():
    probe = LinearProbe(n_features=3, n_classes=2)
    X = [[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]]
    y = [0, 1]
    probe.fit(X, y)
    assert hasattr(probe, "weights")
    assert hasattr(probe, "bias")


# ── ProbingClassifier ───────────────────────────────────────────────────────────


def make_simple_data():
    X = [[1.0, 0.0], [0.0, 1.0], [1.0, 0.1], [0.1, 1.0]]
    y = [0, 1, 0, 1]
    return X, y


def test_probing_classifier_probe_layer_returns_probe_result():
    pc = ProbingClassifier()
    X, y = make_simple_data()
    result = pc.probe_layer(ProbeTask.SENTIMENT, 3, X, y)
    assert isinstance(result, ProbeResult)


def test_probing_classifier_probe_layer_task():
    pc = ProbingClassifier()
    X, y = make_simple_data()
    result = pc.probe_layer(ProbeTask.POS_TAGGING, 0, X, y)
    assert result.task == ProbeTask.POS_TAGGING


def test_probing_classifier_probe_layer_layer_idx():
    pc = ProbingClassifier()
    X, y = make_simple_data()
    result = pc.probe_layer(ProbeTask.ENTITY_TYPE, 5, X, y)
    assert result.layer == 5


def test_probing_classifier_probe_layer_n_samples():
    pc = ProbingClassifier()
    X, y = make_simple_data()
    result = pc.probe_layer(ProbeTask.SYNTAX_DEPTH, 1, X, y)
    assert result.n_samples == len(y)


def test_probing_classifier_probe_layer_chance_level_binary():
    pc = ProbingClassifier()
    X, y = make_simple_data()  # 2 unique labels
    result = pc.probe_layer(ProbeTask.SENTIMENT, 0, X, y)
    assert abs(result.chance_level - 0.5) < 1e-9


def test_probing_classifier_probe_layer_chance_level_multiclass():
    pc = ProbingClassifier()
    X = [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [0.2, 0.8], [0.8, 0.2], [0.3, 0.7]]
    y = [0, 1, 2, 0, 1, 2]
    result = pc.probe_layer(ProbeTask.ENTITY_TYPE, 0, X, y)
    assert abs(result.chance_level - 1 / 3) < 1e-9


def test_probing_classifier_probe_layer_accuracy_range():
    pc = ProbingClassifier()
    X, y = make_simple_data()
    result = pc.probe_layer(ProbeTask.COREFERENCE, 2, X, y)
    assert 0.0 <= result.accuracy <= 1.0


def test_probing_classifier_probe_all_layers_length():
    pc = ProbingClassifier()
    X, y = make_simple_data()
    layers = [X, X, X]
    results = pc.probe_all_layers(ProbeTask.SENTIMENT, layers, y)
    assert len(results) == 3


def test_probing_classifier_probe_all_layers_returns_list():
    pc = ProbingClassifier()
    X, y = make_simple_data()
    results = pc.probe_all_layers(ProbeTask.POS_TAGGING, [X], y)
    assert isinstance(results, list)


def test_probing_classifier_probe_all_layers_each_result():
    pc = ProbingClassifier()
    X, y = make_simple_data()
    results = pc.probe_all_layers(ProbeTask.ENTITY_TYPE, [X, X], y)
    for i, r in enumerate(results):
        assert isinstance(r, ProbeResult)
        assert r.layer == i


def test_probing_classifier_probe_all_layers_zero_layers():
    pc = ProbingClassifier()
    X, y = make_simple_data()
    results = pc.probe_all_layers(ProbeTask.SENTIMENT, [], y)
    assert results == []


def test_probing_classifier_best_layer_returns_highest_accuracy():
    pc = ProbingClassifier()
    results = [
        ProbeResult(ProbeTask.SENTIMENT, 0, 0.6, 10, 0.5),
        ProbeResult(ProbeTask.SENTIMENT, 1, 0.9, 10, 0.5),
        ProbeResult(ProbeTask.SENTIMENT, 2, 0.7, 10, 0.5),
    ]
    best = pc.best_layer(results)
    assert best is not None
    assert best.accuracy == 0.9


def test_probing_classifier_best_layer_none_on_empty():
    pc = ProbingClassifier()
    assert pc.best_layer([]) is None


def test_probing_classifier_best_layer_single():
    pc = ProbingClassifier()
    r = ProbeResult(ProbeTask.POS_TAGGING, 0, 0.75, 50, 0.25)
    best = pc.best_layer([r])
    assert best is r


def test_probing_classifier_init():
    pc = ProbingClassifier()
    assert pc is not None
