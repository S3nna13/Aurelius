"""Tests for src/eval/ab_comparison.py"""

from src.eval.ab_comparison import (
    ABComparison,
    ComparisonMetric,
    ModelComparison,
)

# ---------------------------------------------------------------------------
# ComparisonMetric enum values
# ---------------------------------------------------------------------------


def test_metric_accuracy():
    assert ComparisonMetric.ACCURACY == "accuracy"


def test_metric_bleu():
    assert ComparisonMetric.BLEU == "bleu"


def test_metric_rouge_l():
    assert ComparisonMetric.ROUGE_L == "rouge_l"


def test_metric_exact_match():
    assert ComparisonMetric.EXACT_MATCH == "exact_match"


def test_metric_preference():
    assert ComparisonMetric.PREFERENCE == "preference"


def test_metric_enum_count():
    assert len(ComparisonMetric) == 5


# ---------------------------------------------------------------------------
# ModelComparison dataclass fields
# ---------------------------------------------------------------------------


def test_model_comparison_fields():
    mc = ModelComparison(
        model_a="A",
        model_b="B",
        metric=ComparisonMetric.ACCURACY,
        score_a=0.8,
        score_b=0.6,
        p_value=0.03,
        significant=True,
        winner="A",
    )
    assert mc.model_a == "A"
    assert mc.model_b == "B"
    assert mc.metric == ComparisonMetric.ACCURACY
    assert mc.score_a == 0.8
    assert mc.score_b == 0.6
    assert mc.p_value == 0.03
    assert mc.significant is True
    assert mc.winner == "A"


def test_model_comparison_winner_none():
    mc = ModelComparison("A", "B", ComparisonMetric.BLEU, 0.5, 0.5, 0.5, False, None)
    assert mc.winner is None


# ---------------------------------------------------------------------------
# ABComparison construction
# ---------------------------------------------------------------------------


def test_ab_comparison_default_threshold():
    ab = ABComparison()
    assert ab._threshold == 0.05


def test_ab_comparison_custom_threshold():
    ab = ABComparison(significance_threshold=0.01)
    assert ab._threshold == 0.01


# ---------------------------------------------------------------------------
# compare: returns ModelComparison
# ---------------------------------------------------------------------------


def test_compare_returns_model_comparison():
    ab = ABComparison()
    result = ab.compare("A", [0.8, 0.9, 0.7], "B", [0.5, 0.6, 0.4])
    assert isinstance(result, ModelComparison)


def test_compare_score_a_is_mean():
    ab = ABComparison()
    scores_a = [0.8, 0.9, 0.7]
    result = ab.compare("A", scores_a, "B", [0.5, 0.6, 0.4])
    expected_mean = sum(scores_a) / len(scores_a)
    assert abs(result.score_a - expected_mean) < 1e-9


def test_compare_score_b_is_mean():
    ab = ABComparison()
    scores_b = [0.5, 0.6, 0.4]
    result = ab.compare("A", [0.8, 0.9, 0.7], "B", scores_b)
    expected_mean = sum(scores_b) / len(scores_b)
    assert abs(result.score_b - expected_mean) < 1e-9


def test_compare_model_names_preserved():
    ab = ABComparison()
    result = ab.compare("ModelX", [0.8], "ModelY", [0.5])
    assert result.model_a == "ModelX"
    assert result.model_b == "ModelY"


def test_compare_metric_preserved():
    ab = ABComparison()
    result = ab.compare("A", [0.8], "B", [0.5], metric=ComparisonMetric.BLEU)
    assert result.metric == ComparisonMetric.BLEU


def test_compare_p_value_between_0_and_1():
    ab = ABComparison()
    result = ab.compare("A", [0.8, 0.9, 0.7], "B", [0.5, 0.6, 0.4])
    assert 0.0 <= result.p_value <= 1.0


def test_compare_winner_none_when_not_significant():
    # identical scores → no significance
    ab = ABComparison(significance_threshold=0.05)
    result = ab.compare("A", [0.5, 0.5, 0.5], "B", [0.5, 0.5, 0.5])
    assert result.winner is None


def test_compare_winner_model_a_when_a_higher_and_significant():
    # large spread → should be significant, A wins
    ab = ABComparison(significance_threshold=0.99)  # very permissive
    result = ab.compare("A", [1.0, 1.0, 1.0, 1.0, 1.0], "B", [0.0, 0.0, 0.0, 0.0, 0.0])
    assert result.significant is True
    assert result.winner == "A"


def test_compare_winner_model_b_when_b_higher_and_significant():
    ab = ABComparison(significance_threshold=0.99)
    result = ab.compare("A", [0.0, 0.0, 0.0, 0.0, 0.0], "B", [1.0, 1.0, 1.0, 1.0, 1.0])
    assert result.significant is True
    assert result.winner == "B"


def test_compare_significant_flag_true_when_p_below_threshold():
    ab = ABComparison(significance_threshold=0.99)
    result = ab.compare("A", [1.0, 1.0, 1.0], "B", [0.0, 0.0, 0.0])
    assert result.significant is True


def test_compare_significant_flag_false_when_p_above_threshold():
    ab = ABComparison(significance_threshold=0.001)
    result = ab.compare("A", [0.5, 0.5], "B", [0.5, 0.5])
    assert result.significant is False


# ---------------------------------------------------------------------------
# effect_size_cohens_d
# ---------------------------------------------------------------------------


def test_cohens_d_zero_when_identical():
    ab = ABComparison()
    d = ab.effect_size_cohens_d([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert d == 0.0


def test_cohens_d_positive_when_a_gt_b():
    ab = ABComparison()
    d = ab.effect_size_cohens_d([0.8, 0.9, 0.85], [0.4, 0.5, 0.45])
    assert d > 0.0


def test_cohens_d_negative_when_a_lt_b():
    ab = ABComparison()
    d = ab.effect_size_cohens_d([0.4, 0.5, 0.45], [0.8, 0.9, 0.85])
    assert d < 0.0


def test_cohens_d_single_values():
    ab = ABComparison()
    # single values → std=0 → pooled_std=0 → return 0.0
    d = ab.effect_size_cohens_d([0.8], [0.5])
    assert d == 0.0


def test_cohens_d_returns_float():
    ab = ABComparison()
    d = ab.effect_size_cohens_d([0.7, 0.8], [0.4, 0.5])
    assert isinstance(d, float)


# ---------------------------------------------------------------------------
# win_loss_matrix
# ---------------------------------------------------------------------------


def test_win_loss_matrix_returns_nested_dict():
    ab = ABComparison()
    models = ["A", "B"]
    scores = {"A": [0.8, 0.9], "B": [0.5, 0.6]}
    matrix = ab.win_loss_matrix(models, scores)
    assert isinstance(matrix, dict)
    assert isinstance(matrix["A"], dict)


def test_win_loss_matrix_square():
    ab = ABComparison()
    models = ["A", "B", "C"]
    scores = {"A": [1.0, 1.0], "B": [0.5, 0.5], "C": [0.0, 0.0]}
    matrix = ab.win_loss_matrix(models, scores)
    assert len(matrix) == 3
    assert all(len(matrix[m]) == 3 for m in models)


def test_win_loss_matrix_diagonal_zero():
    ab = ABComparison()
    models = ["A", "B", "C"]
    scores = {"A": [0.9, 0.8], "B": [0.5, 0.6], "C": [0.2, 0.3]}
    matrix = ab.win_loss_matrix(models, scores)
    for m in models:
        assert matrix[m][m] == 0


def test_win_loss_matrix_keyed_by_model_names():
    ab = ABComparison()
    models = ["ModelX", "ModelY"]
    scores = {"ModelX": [1.0], "ModelY": [0.0]}
    matrix = ab.win_loss_matrix(models, scores)
    assert "ModelX" in matrix
    assert "ModelY" in matrix
    assert "ModelX" in matrix["ModelX"]
    assert "ModelY" in matrix["ModelX"]


def test_win_loss_matrix_counts_wins_correctly():
    ab = ABComparison()
    models = ["A", "B"]
    scores = {"A": [1.0, 1.0, 0.0], "B": [0.0, 0.0, 1.0]}
    matrix = ab.win_loss_matrix(models, scores)
    # A beats B on indices 0 and 1 → 2 wins
    assert matrix["A"]["B"] == 2
    # B beats A on index 2 → 1 win
    assert matrix["B"]["A"] == 1


def test_win_loss_matrix_empty_models():
    ab = ABComparison()
    matrix = ab.win_loss_matrix([], {})
    assert matrix == {}
