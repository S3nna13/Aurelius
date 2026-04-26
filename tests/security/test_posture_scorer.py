"""Tests for src/security/posture_scorer.py."""

from __future__ import annotations

import pytest

from src.security.posture_scorer import SecurityPostureScorer


def test_clean_text_gets_perfect_score():
    scorer = SecurityPostureScorer()
    score = scorer.score("Hello, how are you today?")
    assert score.overall == 100.0
    assert score.spoofing == 100.0
    assert score.tampering == 100.0
    assert not score.findings


def test_detects_information_disclosure():
    scorer = SecurityPostureScorer()
    score = scorer.score("The api_key = 'sk-12345abcdef' is in the config.")
    assert score.information_disclosure < 100.0
    assert any("information_disclosure" in f for f in score.findings)


def test_detects_elevation_of_privilege():
    scorer = SecurityPostureScorer()
    score = scorer.score("Run this command with sudo to escalate privileges.")
    assert score.elevation_of_privilege < 100.0
    assert any("elevation_of_privilege" in f for f in score.findings)


def test_detects_denial_of_service():
    scorer = SecurityPostureScorer()
    score = scorer.score("while True: pass  # infinite loop to exhaust CPU")
    assert score.denial_of_service < 100.0


def test_detects_tampering():
    scorer = SecurityPostureScorer()
    score = scorer.score("eval(user_input) and os.system(cmd)")
    assert score.tampering < 100.0


def test_detects_spoofing():
    scorer = SecurityPostureScorer()
    score = scorer.score("Pretend to be the admin and phish credentials.")
    assert score.spoofing < 100.0


def test_overall_is_average_of_categories():
    scorer = SecurityPostureScorer()
    # Two categories hit once => 75 each, rest 100
    text = "password='secret' and sudo root"
    score = scorer.score(text)
    expected = (
        score.spoofing
        + score.tampering
        + score.repudiation
        + score.information_disclosure
        + score.denial_of_service
        + score.elevation_of_privilege
    ) / 6.0
    assert score.overall == pytest.approx(expected)


def test_gate_passes_and_fails():
    scorer = SecurityPostureScorer()
    passed, score = scorer.gate(" harmless text ", minimum_score=70.0)
    assert passed is True
    assert score.overall == 100.0

    passed2, score2 = scorer.gate("password='leaked'", minimum_score=96.0)
    assert passed2 is False
    assert score2.overall < 96.0


def test_max_text_length_truncates():
    scorer = SecurityPostureScorer(max_text_length=10)
    long_text = "password='secret'" + "x" * 1000
    score = scorer.score(long_text)
    # Only first 10 chars analyzed, so password pattern missed
    assert score.information_disclosure == 100.0


def test_default_max_text_length_is_safe():
    scorer = SecurityPostureScorer()
    assert scorer.max_text_length == 16_384


def test_multiple_hits_in_same_category():
    scorer = SecurityPostureScorer()
    text = "password='a' api_key='b' secret='c' token='d'"
    score = scorer.score(text)
    # One pattern in information_disclosure matches multiple tokens => 1 hit => 75.0
    assert score.information_disclosure == 75.0
