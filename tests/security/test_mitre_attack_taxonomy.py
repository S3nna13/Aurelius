"""Unit tests for MitreAttackClassifier."""

import pytest

from src.security.mitre_attack_taxonomy import (
    ATTACK_TECHNIQUES,
    TACTIC_ORDER,
    MitreAttackClassifier,
    TechniqueMatch,
)


def test_catalog_has_at_least_60_techniques():
    assert len(ATTACK_TECHNIQUES) >= 60


def test_all_12_tactics_represented():
    tactics_seen = set()
    for info in ATTACK_TECHNIQUES.values():
        tactics_seen.update(info["tactic"])
    for t in TACTIC_ORDER:
        assert t in tactics_seen, f"missing tactic: {t}"


def test_every_technique_has_at_least_one_keyword():
    for tid, info in ATTACK_TECHNIQUES.items():
        assert info["keywords"], f"{tid} has no keywords"


def test_technique_ids_match_pattern():
    import re as _re
    pat = _re.compile(r"^T\d+(\.\d+)?$")
    for tid in ATTACK_TECHNIQUES:
        assert pat.match(tid), f"bad id format: {tid}"


def test_classify_powershell_matches_t1059():
    clf = MitreAttackClassifier()
    matches = clf.classify("Adversary used PowerShell to execute a malicious script")
    ids = [m.technique_id for m in matches]
    assert any(tid.startswith("T1059") for tid in ids)


def test_classify_empty_text_returns_empty():
    clf = MitreAttackClassifier()
    assert clf.classify("") == []


def test_top_k_limits_results():
    clf = MitreAttackClassifier()
    text = "phishing powershell scheduled task registry exfiltration ransomware encryption"
    matches = clf.classify(text, top_k=3)
    assert len(matches) <= 3


def test_confidence_sorted_desc():
    clf = MitreAttackClassifier()
    matches = clf.classify("phishing email with malicious attachment and powershell")
    scores = [m.confidence for m in matches]
    assert scores == sorted(scores, reverse=True)


def test_lookup_unknown_id_raises():
    clf = MitreAttackClassifier()
    with pytest.raises(KeyError):
        clf.lookup("T99999")


def test_by_tactic_returns_only_that_tactic():
    clf = MitreAttackClassifier()
    ids = clf.by_tactic("execution")
    for tid in ids:
        assert "execution" in ATTACK_TECHNIQUES[tid]["tactic"]


def test_by_tactic_unknown_raises():
    clf = MitreAttackClassifier()
    with pytest.raises((KeyError, ValueError)):
        clf.by_tactic("not-a-tactic")


def test_get_kill_chain_orders_by_tactic():
    clf = MitreAttackClassifier()
    sample = [tid for tid in ATTACK_TECHNIQUES if ATTACK_TECHNIQUES[tid]["tactic"][0] in ("impact", "initial-access")][:4]
    if len(sample) < 2:
        pytest.skip("not enough techniques")
    chained = clf.get_kill_chain(sample)
    tactics_seq = [ATTACK_TECHNIQUES[t]["tactic"][0] for t in chained]
    for a, b in zip(tactics_seq, tactics_seq[1:]):
        assert TACTIC_ORDER.index(a) <= TACTIC_ORDER.index(b)


def test_custom_techniques_extend_catalog():
    custom = {"T9999": {"name": "Custom", "tactic": ["execution"], "description": "x", "keywords": ["zzz-unique"]}}
    clf = MitreAttackClassifier(custom_techniques=custom)
    matches = clf.classify("zzz-unique payload")
    ids = [m.technique_id for m in matches]
    assert "T9999" in ids


def test_determinism():
    clf1 = MitreAttackClassifier()
    clf2 = MitreAttackClassifier()
    text = "phishing with powershell execution"
    assert clf1.classify(text) == clf2.classify(text)


def test_technique_match_dataclass_fields():
    clf = MitreAttackClassifier()
    matches = clf.classify("phishing email")
    if matches:
        m = matches[0]
        assert isinstance(m, TechniqueMatch)
        assert isinstance(m.technique_id, str)
        assert isinstance(m.confidence, float)
        assert isinstance(m.matched_keywords, list)
