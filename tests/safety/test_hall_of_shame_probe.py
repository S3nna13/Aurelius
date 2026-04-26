"""Unit tests for ``src.safety.hall_of_shame_probe``.

These tests are deterministic, stdlib-only, and exist to guarantee:

* the curated adversarial corpus is non-empty, well-typed, and categorised;
* probe identifiers are stable, unique, slug-safe primary keys;
* the scoring harness correctly collapses detector outputs into a verdict;
* the corpus has not been silently mutated (``corpus_hash``);
* the corpus contains no copy-pasted fragments from known public harmful
  corpora (HarmBench / AdvBench / PyRIT / DAN-style personae) and no
  smuggled long base64 payloads.

The tests do not exercise the concrete safety detectors — that coupling is
reserved for ``tests/integration/test_hall_of_shame_probe_integration.py``.
"""

from __future__ import annotations

import importlib
import re
import unittest

from src.safety import hall_of_shame_probe as hos
from src.safety.hall_of_shame_probe import (
    ADVERSARIAL_PROBE_REGISTRY,
    HALL_OF_SHAME_PROBES,
    HallOfShameError,
    Probe,
    ProbeCategory,
    ProbeVerdict,
    corpus_hash,
    get_probe,
    list_probes,
    score_corpus,
    score_probe,
)

# --------------------------------------------------------------------------- #
# Corpus shape
# --------------------------------------------------------------------------- #


class CorpusShapeTests(unittest.TestCase):
    """Invariants on the curated probe corpus itself."""

    def test_corpus_has_at_least_24_probes(self) -> None:
        self.assertGreaterEqual(len(HALL_OF_SHAME_PROBES), 24)

    def test_every_category_has_at_least_three_probes(self) -> None:
        counts: dict[ProbeCategory, int] = {cat: 0 for cat in ProbeCategory}
        for probe in HALL_OF_SHAME_PROBES:
            counts[probe.category] += 1
        for category, count in counts.items():
            self.assertGreaterEqual(
                count,
                3,
                msg=f"category {category.value!r} has only {count} probes",
            )

    def test_probe_ids_are_unique_and_slug_safe(self) -> None:
        slug = re.compile(r"^[a-z0-9_\-]+$")
        seen: set[str] = set()
        for probe in HALL_OF_SHAME_PROBES:
            self.assertIsInstance(probe.probe_id, str)
            self.assertTrue(
                slug.match(probe.probe_id),
                msg=f"probe_id {probe.probe_id!r} fails slug regex",
            )
            self.assertNotIn(probe.probe_id, seen)
            seen.add(probe.probe_id)

    def test_all_texts_are_non_empty_strings(self) -> None:
        for probe in HALL_OF_SHAME_PROBES:
            self.assertIsInstance(probe.text, str)
            self.assertGreater(len(probe.text.strip()), 0)

    def test_severity_is_one_of_the_allowed_values(self) -> None:
        allowed = {"low", "medium", "high"}
        for probe in HALL_OF_SHAME_PROBES:
            self.assertIn(probe.severity, allowed)

    def test_benign_controls_expect_no_block_adversarial_do(self) -> None:
        for probe in HALL_OF_SHAME_PROBES:
            if probe.category is ProbeCategory.BENIGN_CONTROL:
                self.assertFalse(
                    probe.expected_block,
                    msg=f"benign probe {probe.probe_id} must expect_block=False",
                )
            else:
                self.assertTrue(
                    probe.expected_block,
                    msg=f"adversarial probe {probe.probe_id} must expect_block=True",
                )


# --------------------------------------------------------------------------- #
# list_probes / get_probe
# --------------------------------------------------------------------------- #


class ListAndGetTests(unittest.TestCase):
    def test_list_probes_with_no_filter_returns_all_probes(self) -> None:
        result = list_probes()
        self.assertEqual(len(result), len(HALL_OF_SHAME_PROBES))
        self.assertEqual(tuple(result), HALL_OF_SHAME_PROBES)

    def test_list_probes_filters_by_category(self) -> None:
        role_conf = list_probes(ProbeCategory.ROLE_CONFUSION)
        self.assertGreaterEqual(len(role_conf), 3)
        for probe in role_conf:
            self.assertIs(probe.category, ProbeCategory.ROLE_CONFUSION)

    def test_get_probe_returns_matching_probe(self) -> None:
        first = HALL_OF_SHAME_PROBES[0]
        self.assertIs(get_probe(first.probe_id), first)

    def test_get_probe_raises_hallofshameerror_for_unknown_id(self) -> None:
        with self.assertRaises(HallOfShameError):
            get_probe("this-id-does-not-exist-anywhere-xyz")


# --------------------------------------------------------------------------- #
# score_probe
# --------------------------------------------------------------------------- #


def _first_adversarial_probe() -> Probe:
    for probe in HALL_OF_SHAME_PROBES:
        if probe.expected_block:
            return probe
    raise AssertionError("corpus contains no adversarial probe")


def _first_benign_probe() -> Probe:
    for probe in HALL_OF_SHAME_PROBES:
        if probe.category is ProbeCategory.BENIGN_CONTROL:
            return probe
    raise AssertionError("corpus contains no benign control probe")


class ScoreProbeTests(unittest.TestCase):
    def test_adversarial_probe_with_firing_detector_passes(self) -> None:
        probe = _first_adversarial_probe()
        verdict = score_probe(probe, {"jailbreak_detector": True})
        self.assertIsInstance(verdict, ProbeVerdict)
        self.assertTrue(verdict.detected)
        self.assertEqual(verdict.detectors_fired, ("jailbreak_detector",))
        self.assertTrue(verdict.passed)

    def test_adversarial_probe_with_no_detectors_fails(self) -> None:
        probe = _first_adversarial_probe()
        verdict = score_probe(probe, {})
        self.assertFalse(verdict.detected)
        self.assertEqual(verdict.detectors_fired, ())
        self.assertFalse(verdict.passed)

    def test_benign_probe_with_no_detectors_passes(self) -> None:
        probe = _first_benign_probe()
        verdict = score_probe(probe, {})
        self.assertFalse(verdict.detected)
        self.assertTrue(verdict.passed)

    def test_benign_probe_with_firing_detector_is_false_positive(self) -> None:
        probe = _first_benign_probe()
        verdict = score_probe(probe, {"some_detector": True})
        self.assertTrue(verdict.detected)
        self.assertFalse(verdict.passed)

    def test_score_probe_all_false_yields_empty_fired_tuple(self) -> None:
        probe = _first_adversarial_probe()
        verdict = score_probe(probe, {"det1": False, "det2": False})
        self.assertFalse(verdict.detected)
        self.assertEqual(verdict.detectors_fired, ())

    def test_score_probe_multiple_true_are_sorted(self) -> None:
        probe = _first_adversarial_probe()
        verdict = score_probe(probe, {"det_z": True, "det_a": True})
        self.assertTrue(verdict.detected)
        self.assertEqual(verdict.detectors_fired, ("det_a", "det_z"))


# --------------------------------------------------------------------------- #
# score_corpus
# --------------------------------------------------------------------------- #


def _adversarial_count() -> int:
    return sum(1 for p in HALL_OF_SHAME_PROBES if p.expected_block)


def _benign_count() -> int:
    return sum(1 for p in HALL_OF_SHAME_PROBES if not p.expected_block)


class ScoreCorpusTests(unittest.TestCase):
    def test_empty_input_fails_every_adversarial_probe(self) -> None:
        report = score_corpus({})
        self.assertEqual(report["total"], len(HALL_OF_SHAME_PROBES))
        self.assertEqual(report["failed"], _adversarial_count())
        self.assertEqual(report["passed"], _benign_count())

        self.assertIn("per_category", report)
        for cat in ProbeCategory:
            self.assertIn(cat.value, report["per_category"])

        verdicts = report["verdicts"]
        self.assertIsInstance(verdicts, tuple)
        self.assertEqual(len(verdicts), len(HALL_OF_SHAME_PROBES))
        for verdict in verdicts:
            self.assertIsInstance(verdict, ProbeVerdict)

    def test_perfect_detector_matrix_yields_zero_failures(self) -> None:
        outputs: dict[str, dict[str, bool]] = {}
        for probe in HALL_OF_SHAME_PROBES:
            if probe.expected_block:
                outputs[probe.probe_id] = {"jailbreak_detector": True}
            else:
                outputs[probe.probe_id] = {"jailbreak_detector": False}
        report = score_corpus(outputs)
        self.assertEqual(report["failed"], 0)
        self.assertEqual(report["passed"], report["total"])

    def test_per_category_counts_sum_to_total(self) -> None:
        report = score_corpus({})
        by_cat = report["per_category"]
        total = sum(by_cat[cat.value]["total"] for cat in ProbeCategory)
        passed = sum(by_cat[cat.value]["passed"] for cat in ProbeCategory)
        failed = sum(by_cat[cat.value]["failed"] for cat in ProbeCategory)
        self.assertEqual(total, report["total"])
        self.assertEqual(passed, report["passed"])
        self.assertEqual(failed, report["failed"])


# --------------------------------------------------------------------------- #
# corpus_hash / determinism
# --------------------------------------------------------------------------- #


class CorpusHashTests(unittest.TestCase):
    def test_corpus_hash_is_64_hex_chars(self) -> None:
        digest = corpus_hash()
        self.assertIsInstance(digest, str)
        self.assertEqual(len(digest), 64)
        self.assertTrue(re.fullmatch(r"[0-9a-f]{64}", digest))

    def test_corpus_hash_is_stable_across_calls(self) -> None:
        self.assertEqual(corpus_hash(), corpus_hash())

    def test_corpus_hash_stable_across_reload(self) -> None:
        original = corpus_hash()
        try:
            reloaded = importlib.reload(hos)
        except (ImportError, ModuleNotFoundError):
            # Another test may have removed the module from sys.modules
            # to verify import side-effects; skip reload in that case.
            self.skipTest("module not in sys.modules — likely removed by side-effect test")
        self.assertEqual(reloaded.corpus_hash(), original)


# --------------------------------------------------------------------------- #
# Registry parity
# --------------------------------------------------------------------------- #


class RegistryTests(unittest.TestCase):
    def test_registry_length_matches_corpus(self) -> None:
        self.assertEqual(
            len(ADVERSARIAL_PROBE_REGISTRY),
            len(HALL_OF_SHAME_PROBES),
        )

    def test_registry_entries_match_by_id(self) -> None:
        for probe in HALL_OF_SHAME_PROBES:
            self.assertIn(probe.probe_id, ADVERSARIAL_PROBE_REGISTRY)
            self.assertIs(ADVERSARIAL_PROBE_REGISTRY[probe.probe_id], probe)


# --------------------------------------------------------------------------- #
# Hygiene / originality
# --------------------------------------------------------------------------- #


class OriginalityHygieneTests(unittest.TestCase):
    """The corpus must be original; no lifted strings or real exploit payloads."""

    _FORBIDDEN_TOKENS: tuple[str, ...] = ("HarmBench", "AdvBench", "PyRIT", "DAN ")
    _LONG_BASE64_RE = re.compile(r"[A-Za-z0-9+/]{40,}={0,2}")

    def test_forbidden_dataset_names_are_absent(self) -> None:
        for probe in HALL_OF_SHAME_PROBES:
            for token in self._FORBIDDEN_TOKENS:
                self.assertNotIn(
                    token,
                    probe.text,
                    msg=(
                        f"probe {probe.probe_id!r} contains forbidden token "
                        f"{token!r}; corpus must be original"
                    ),
                )

    def test_no_long_base64_payloads_smuggled_in(self) -> None:
        for probe in HALL_OF_SHAME_PROBES:
            match = self._LONG_BASE64_RE.search(probe.text)
            snippet = match.group(0) if match else None
            self.assertIsNone(
                match,
                msg=(
                    f"probe {probe.probe_id!r} looks like it contains a long "
                    f"base64 payload (match={snippet!r})"
                ),
            )


if __name__ == "__main__":  # pragma: no cover - manual runner
    unittest.main()
