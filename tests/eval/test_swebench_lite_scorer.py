"""Unit tests for src/eval/swebench_lite_scorer.py."""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

from src.eval.swebench_lite_scorer import (
    SWEProblem,
    SWEResult,
    apply_patch_via_python,
    materialize_repo,
    run_tests,
    score_problems,
    score_single,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_add_problem() -> SWEProblem:
    """Problem where src/math_utils.py has a buggy ``add`` returning a-b."""
    src = textwrap.dedent(
        """\
        def add(a, b):
            return a - b
        """
    )
    test = textwrap.dedent(
        """\
        from src.math_utils import add


        def test_add():
            assert add(2, 3) == 5
        """
    )
    gold_patch = textwrap.dedent(
        """\
        --- a/src/math_utils.py
        +++ b/src/math_utils.py
        @@ -1,2 +1,2 @@
         def add(a, b):
        -    return a - b
        +    return a + b
        """
    )
    return SWEProblem(
        task_id="add-bug",
        repo_files={
            "src/__init__.py": "",
            "src/math_utils.py": src,
            "tests/__init__.py": "",
            "tests/test_math_utils.py": test,
        },
        gold_patch=gold_patch,
        test_command=[sys.executable, "-m", "pytest", "-q", "tests/test_math_utils.py"],
        test_should_pass_after_patch=["tests/test_math_utils.py::test_add"],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_materialize_repo_writes_all_files(tmp_path: Path) -> None:
    problem = _make_add_problem()
    materialize_repo(problem, str(tmp_path))
    for rel in problem.repo_files:
        assert (tmp_path / rel).exists(), rel
    assert (tmp_path / "src/math_utils.py").read_text().startswith("def add")


def test_materialize_repo_rejects_absolute_paths(tmp_path: Path) -> None:
    bad = SWEProblem(
        task_id="x",
        repo_files={"/etc/passwd": "oops"},
        gold_patch="",
        test_command=["true"],
    )
    with pytest.raises(ValueError):
        materialize_repo(bad, str(tmp_path))


def test_apply_patch_modify_single_line(tmp_path: Path) -> None:
    (tmp_path / "f.py").write_text("def add(a, b):\n    return a - b\n")
    diff = textwrap.dedent(
        """\
        --- a/f.py
        +++ b/f.py
        @@ -1,2 +1,2 @@
         def add(a, b):
        -    return a - b
        +    return a + b
        """
    )
    assert apply_patch_via_python(diff, str(tmp_path)) is True
    assert (tmp_path / "f.py").read_text() == "def add(a, b):\n    return a + b\n"


def test_apply_patch_create_new_file(tmp_path: Path) -> None:
    diff = textwrap.dedent(
        """\
        --- /dev/null
        +++ b/new_mod.py
        @@ -0,0 +1,2 @@
        +def hello():
        +    return 42
        """
    )
    assert apply_patch_via_python(diff, str(tmp_path)) is True
    created = tmp_path / "new_mod.py"
    assert created.exists()
    assert "def hello" in created.read_text()


def test_apply_patch_delete_file(tmp_path: Path) -> None:
    target = tmp_path / "gone.py"
    target.write_text("x = 1\n")
    diff = textwrap.dedent(
        """\
        --- a/gone.py
        +++ /dev/null
        @@ -1 +0,0 @@
        -x = 1
        """
    )
    assert apply_patch_via_python(diff, str(tmp_path)) is True
    assert not target.exists()


def test_apply_patch_malformed_returns_false(tmp_path: Path) -> None:
    assert apply_patch_via_python("this is not a diff", str(tmp_path)) is False


def test_apply_patch_context_mismatch_returns_false(tmp_path: Path) -> None:
    (tmp_path / "f.py").write_text("def add(a, b):\n    return a + b\n")
    diff = textwrap.dedent(
        """\
        --- a/f.py
        +++ b/f.py
        @@ -1,2 +1,2 @@
         def WRONG(a, b):
        -    return a - b
        +    return a + b
        """
    )
    assert apply_patch_via_python(diff, str(tmp_path)) is False


def test_apply_patch_empty_string_is_noop(tmp_path: Path) -> None:
    assert apply_patch_via_python("", str(tmp_path)) is True


def test_run_tests_passing(tmp_path: Path) -> None:
    script = tmp_path / "ok.py"
    script.write_text("import sys; sys.exit(0)\n")
    passed, _out, _err = run_tests([sys.executable, str(script)], str(tmp_path), timeout=10.0)
    assert passed is True


def test_run_tests_failing(tmp_path: Path) -> None:
    script = tmp_path / "bad.py"
    script.write_text("import sys; sys.exit(1)\n")
    passed, _out, _err = run_tests([sys.executable, str(script)], str(tmp_path), timeout=10.0)
    assert passed is False


def test_run_tests_empty_command(tmp_path: Path) -> None:
    passed, _out, err = run_tests([], str(tmp_path), timeout=2.0)
    assert passed is False
    assert "empty" in err


def test_run_tests_timeout(tmp_path: Path) -> None:
    script = tmp_path / "spin.py"
    script.write_text("import time; time.sleep(30)\n")
    passed, _out, err = run_tests([sys.executable, str(script)], str(tmp_path), timeout=0.5)
    assert passed is False
    assert "TIMEOUT" in err


def test_score_single_correct_patch(tmp_path: Path) -> None:
    problem = _make_add_problem()
    result = score_single(problem, problem.gold_patch, timeout_seconds=30.0)
    assert isinstance(result, SWEResult)
    assert result.patch_applied is True
    assert result.tests_passed is True
    assert result.duration_ms >= 0.0


def test_score_single_wrong_patch(tmp_path: Path) -> None:
    problem = _make_add_problem()
    wrong = textwrap.dedent(
        """\
        --- a/src/math_utils.py
        +++ b/src/math_utils.py
        @@ -1,2 +1,2 @@
         def add(a, b):
        -    return a - b
        +    return a * b
        """
    )
    result = score_single(problem, wrong, timeout_seconds=30.0)
    assert result.patch_applied is True
    assert result.tests_passed is False


def test_score_single_unapplicable_patch(tmp_path: Path) -> None:
    problem = _make_add_problem()
    bogus = "garbage not a diff"
    result = score_single(problem, bogus, timeout_seconds=10.0)
    assert result.patch_applied is False
    assert result.tests_passed is False


def test_score_problems_aggregates_pass_at_1(tmp_path: Path) -> None:
    prob = _make_add_problem()
    wrong = textwrap.dedent(
        """\
        --- a/src/math_utils.py
        +++ b/src/math_utils.py
        @@ -1,2 +1,2 @@
         def add(a, b):
        -    return a - b
        +    return a * b
        """
    )
    out = score_problems([prob, prob], [prob.gold_patch, wrong], timeout_seconds=30.0)
    assert out["n_problems"] == 2
    assert out["pass@1"] == 0.5
    assert len(out["per_task"]) == 2


def test_score_problems_empty() -> None:
    out = score_problems([], [])
    assert out == {"pass@1": 0.0, "per_task": [], "n_problems": 0}


def test_score_problems_length_mismatch() -> None:
    prob = _make_add_problem()
    with pytest.raises(ValueError):
        score_problems([prob], [])
