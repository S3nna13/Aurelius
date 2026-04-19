"""Integration tests: SWE-bench-lite registry + end-to-end scoring."""

from __future__ import annotations

import sys
import textwrap

from src import eval as eval_pkg


def test_metric_registry_has_swebench_lite() -> None:
    assert "swebench_lite" in eval_pkg.METRIC_REGISTRY


def test_benchmark_registry_has_swebench_lite() -> None:
    assert "swebench_lite" in eval_pkg.BENCHMARK_REGISTRY


def test_prior_registry_entries_intact() -> None:
    for key in ("niah", "ruler", "humaneval", "mbpp"):
        assert key in eval_pkg.METRIC_REGISTRY, f"METRIC_REGISTRY missing {key}"
        assert key in eval_pkg.BENCHMARK_REGISTRY, f"BENCHMARK_REGISTRY missing {key}"


def test_end_to_end_single_synthetic_problem() -> None:
    SWEProblem = eval_pkg.SWEProblem
    score_problems = eval_pkg.swebench_score_problems

    src = "def add(a, b):\n    return a - b\n"
    tests = textwrap.dedent(
        """\
        from pkg.math_utils import add


        def test_add():
            assert add(2, 3) == 5
        """
    )
    gold = textwrap.dedent(
        """\
        --- a/pkg/math_utils.py
        +++ b/pkg/math_utils.py
        @@ -1,2 +1,2 @@
         def add(a, b):
        -    return a - b
        +    return a + b
        """
    )
    problem = SWEProblem(
        task_id="int-add",
        repo_files={
            "pkg/__init__.py": "",
            "pkg/math_utils.py": src,
            "tests/__init__.py": "",
            "tests/test_add.py": tests,
        },
        gold_patch=gold,
        test_command=[sys.executable, "-m", "pytest", "-q", "tests/test_add.py"],
        test_should_pass_after_patch=["tests/test_add.py::test_add"],
    )
    out = score_problems([problem], [gold], timeout_seconds=30.0)
    assert out["n_problems"] == 1
    assert out["pass@1"] == 1.0
    assert out["per_task"][0]["patch_applied"] is True
    assert out["per_task"][0]["tests_passed"] is True
