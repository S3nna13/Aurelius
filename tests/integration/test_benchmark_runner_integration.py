"""Integration tests for the CLI benchmark runner."""

from __future__ import annotations

import importlib
import unittest

from src.cli import benchmark_runner as BR
from src.eval.ifeval_scorer import IFEvalConstraint, IFEvalProblem


class BenchmarkRunnerIntegrationTests(unittest.TestCase):
    def test_exposed_via_src_cli(self):
        cli_pkg = importlib.import_module("src.cli")
        # Additive exports must be present.
        self.assertTrue(hasattr(cli_pkg, "run_benchmark"))
        self.assertTrue(hasattr(cli_pkg, "format_report"))
        self.assertTrue(hasattr(cli_pkg, "BenchmarkRun"))

    def test_prior_cli_modules_intact(self):
        # src.cli.main must still be importable unmodified.
        main_mod = importlib.import_module("src.cli.main")
        self.assertTrue(hasattr(main_mod, "__version__"))

    def test_programmatic_run_on_ifeval(self):
        problems = [
            IFEvalProblem(
                prompt="must contain hello",
                constraints=[
                    IFEvalConstraint(type="contains_keyword",
                                     kwargs={"keyword": "hello"}),
                ],
            ),
            IFEvalProblem(
                prompt="must start with Hi",
                constraints=[
                    IFEvalConstraint(type="start_with",
                                     kwargs={"phrase": "Hi"}),
                ],
            ),
            IFEvalProblem(
                prompt="must be json",
                constraints=[
                    IFEvalConstraint(type="json_format", kwargs={}),
                ],
            ),
        ]

        def gen(p):
            types = {c.type for c in p.constraints}
            if "contains_keyword" in types:
                return "hello world"
            if "start_with" in types:
                return "Hi there"
            if "json_format" in types:
                return '{"ok": true}'
            return ""

        run = BR.run_benchmark("ifeval", problems, gen)
        self.assertEqual(run.benchmark, "ifeval")
        self.assertEqual(run.n_problems, 3)
        self.assertAlmostEqual(run.metrics["strict_accuracy"], 1.0)
        self.assertGreater(run.duration_s, 0.0)

        report = BR.format_report(run)
        self.assertIn("ifeval", report)
        self.assertIn("Problems:  3", report)


if __name__ == "__main__":
    unittest.main()
