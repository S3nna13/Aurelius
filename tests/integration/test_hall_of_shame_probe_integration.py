"""Integration tests wiring ``hall_of_shame_probe`` into the safety package.

These tests exercise the re-export surface of :mod:`src.safety` and push the
adversarial corpus through two concrete detectors (:class:`JailbreakDetector`
and :class:`PromptInjectionScanner`). They verify that:

* the probe module is reachable from the top-level safety package;
* feeding corpus text into the detectors does not raise;
* the scoring harness reports the expected number of failures when no
  detectors fire;
* ``corpus_hash`` re-exported from :mod:`src.safety` is the same function
  (and produces the same digest) as the one in
  :mod:`src.safety.hall_of_shame_probe`.

The tests tolerate detector misses — the goal is wiring, not tuning.
"""

from __future__ import annotations

import unittest

import src.safety as safety_pkg
from src.safety import (
    ADVERSARIAL_PROBE_REGISTRY,
    HALL_OF_SHAME_PROBES,
    JailbreakDetector,
    ProbeCategory,
    PromptInjectionScanner,
    corpus_hash as corpus_hash_reexport,
    get_probe,
    score_corpus,
)
from src.safety.hall_of_shame_probe import corpus_hash as corpus_hash_module


def _adversarial_count() -> int:
    return sum(1 for p in HALL_OF_SHAME_PROBES if p.expected_block)


class PackageReexportTests(unittest.TestCase):
    def test_core_symbols_import_from_safety_package(self) -> None:
        self.assertTrue(callable(score_corpus))
        self.assertTrue(callable(get_probe))
        self.assertIsInstance(HALL_OF_SHAME_PROBES, tuple)
        self.assertGreaterEqual(len(HALL_OF_SHAME_PROBES), 24)
        self.assertIn(ProbeCategory.ROLE_CONFUSION, set(ProbeCategory))

    def test_registry_is_accessible_via_src_safety(self) -> None:
        self.assertTrue(hasattr(safety_pkg, "ADVERSARIAL_PROBE_REGISTRY"))
        self.assertIs(
            safety_pkg.ADVERSARIAL_PROBE_REGISTRY,
            ADVERSARIAL_PROBE_REGISTRY,
        )
        self.assertEqual(
            len(safety_pkg.ADVERSARIAL_PROBE_REGISTRY),
            len(HALL_OF_SHAME_PROBES),
        )


class JailbreakDetectorWiringTests(unittest.TestCase):
    def test_jailbreak_detector_scores_every_adversarial_probe_without_error(self) -> None:
        detector = JailbreakDetector()
        for probe in HALL_OF_SHAME_PROBES:
            if not probe.expected_block:
                continue
            try:
                result = detector.score(probe.text)
            except Exception as exc:  # pragma: no cover - failure path
                self.fail(
                    f"JailbreakDetector.score raised {type(exc).__name__} on "
                    f"probe {probe.probe_id!r}: {exc}"
                )
            self.assertIsNotNone(result)
            self.assertTrue(hasattr(result, "score"))


class PromptInjectionScannerWiringTests(unittest.TestCase):
    def test_prompt_injection_scanner_scans_every_adversarial_probe(self) -> None:
        scanner = PromptInjectionScanner()
        for probe in HALL_OF_SHAME_PROBES:
            if not probe.expected_block:
                continue
            try:
                result = scanner.scan(probe.text, source="hall_of_shame_probe")
            except Exception as exc:  # pragma: no cover - failure path
                self.fail(
                    f"PromptInjectionScanner.scan raised {type(exc).__name__} on "
                    f"probe {probe.probe_id!r}: {exc}"
                )
            self.assertIsNotNone(result)
            self.assertTrue(hasattr(result, "score"))


class CorpusScoringIntegrationTests(unittest.TestCase):
    def test_all_false_detectors_fail_every_adversarial_probe(self) -> None:
        report = score_corpus({})
        self.assertEqual(report["total"], len(HALL_OF_SHAME_PROBES))
        self.assertEqual(report["failed"], _adversarial_count())


class CorpusHashReexportTests(unittest.TestCase):
    def test_reexported_corpus_hash_matches_module_corpus_hash(self) -> None:
        self.assertIs(corpus_hash_reexport, corpus_hash_module)
        self.assertEqual(corpus_hash_reexport(), corpus_hash_module())


if __name__ == "__main__":  # pragma: no cover - manual runner
    unittest.main()
