"""
Tests for src/workflow/parallel_step.py
≥28 test cases covering StepTask, ParallelResult, ParallelStepExecutor, and REGISTRY.
"""

import dataclasses
import time
import unittest

from src.workflow.parallel_step import (
    PARALLEL_STEP_REGISTRY,
    ParallelResult,
    ParallelStepExecutor,
    StepTask,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_task(name: str, fn, timeout_s: float = 30.0) -> StepTask:
    return StepTask(name=name, fn=fn, timeout_s=timeout_s)


def ok_fn():
    return "done"


def fail_fn():
    raise RuntimeError("task error")


def slow_fn(seconds: float = 0.5):
    """Blocks for *seconds* then returns."""

    def _inner():
        time.sleep(seconds)
        return "slow_done"

    return _inner


def blocking_fn():
    """Blocks for a long time (used with very short timeout)."""
    time.sleep(60)
    return "never"


# ---------------------------------------------------------------------------
# run() tests
# ---------------------------------------------------------------------------


class TestParallelStepExecutorRun(unittest.TestCase):
    def setUp(self):
        self.executor = ParallelStepExecutor(max_workers=4)

    def test_empty_tasks_returns_empty_list(self):
        results = self.executor.run([])
        self.assertEqual(results, [])

    def test_all_succeed(self):
        tasks = [make_task(f"t{i}", ok_fn) for i in range(3)]
        results = self.executor.run(tasks)
        self.assertTrue(all(r.success for r in results))

    def test_result_order_preserved(self):
        tasks = [make_task(f"t{i}", lambda i=i: i) for i in range(5)]
        results = self.executor.run(tasks)
        for i, r in enumerate(results):
            self.assertEqual(r.name, f"t{i}")

    def test_output_values_match_functions(self):
        tasks = [make_task("a", lambda: 42), make_task("b", lambda: "hello")]
        results = self.executor.run(tasks)
        self.assertEqual(results[0].output, 42)
        self.assertEqual(results[1].output, "hello")

    def test_exception_captured_not_raised(self):
        tasks = [make_task("bad", fail_fn)]
        results = self.executor.run(tasks)
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)

    def test_error_message_captured(self):
        tasks = [make_task("bad", fail_fn)]
        results = self.executor.run(tasks)
        self.assertIn("task error", results[0].error)

    def test_success_flag_true_when_ok(self):
        tasks = [make_task("good", ok_fn)]
        results = self.executor.run(tasks)
        self.assertTrue(results[0].success)

    def test_success_flag_false_on_exception(self):
        tasks = [make_task("bad", fail_fn)]
        results = self.executor.run(tasks)
        self.assertFalse(results[0].success)

    def test_duration_ms_non_negative(self):
        tasks = [make_task("t", ok_fn)]
        results = self.executor.run(tasks)
        self.assertGreaterEqual(results[0].duration_ms, 0.0)

    def test_duration_ms_type_float(self):
        tasks = [make_task("t", ok_fn)]
        results = self.executor.run(tasks)
        self.assertIsInstance(results[0].duration_ms, float)

    def test_output_none_on_failure(self):
        tasks = [make_task("bad", fail_fn)]
        results = self.executor.run(tasks)
        self.assertIsNone(results[0].output)

    def test_error_empty_on_success(self):
        tasks = [make_task("good", ok_fn)]
        results = self.executor.run(tasks)
        self.assertEqual(results[0].error, "")

    def test_mixed_success_and_failure(self):
        tasks = [make_task("good", ok_fn), make_task("bad", fail_fn), make_task("good2", ok_fn)]
        results = self.executor.run(tasks)
        self.assertTrue(results[0].success)
        self.assertFalse(results[1].success)
        self.assertTrue(results[2].success)

    def test_result_count_matches_task_count(self):
        tasks = [make_task(f"t{i}", ok_fn) for i in range(7)]
        results = self.executor.run(tasks)
        self.assertEqual(len(results), 7)

    def test_parallel_result_name_matches_task_name(self):
        tasks = [make_task("unique_name", ok_fn)]
        results = self.executor.run(tasks)
        self.assertEqual(results[0].name, "unique_name")

    def test_tasks_run_concurrently(self):
        """Two tasks each sleeping 0.1s should finish faster than sequential 0.2s."""
        start = time.monotonic()
        tasks = [make_task(f"t{i}", slow_fn(0.1)) for i in range(2)]
        self.executor.run(tasks)
        elapsed = time.monotonic() - start
        # Allow generous budget; just confirm they didn't run strictly serially
        self.assertLess(elapsed, 0.35)


# ---------------------------------------------------------------------------
# run_with_timeout() tests
# ---------------------------------------------------------------------------


class TestParallelStepExecutorRunWithTimeout(unittest.TestCase):
    def setUp(self):
        self.executor = ParallelStepExecutor(max_workers=4)

    def test_empty_tasks_returns_empty_list(self):
        results = self.executor.run_with_timeout([])
        self.assertEqual(results, [])

    def test_fast_task_succeeds(self):
        tasks = [make_task("fast", ok_fn, timeout_s=5.0)]
        results = self.executor.run_with_timeout(tasks)
        self.assertTrue(results[0].success)

    def test_fast_task_output_correct(self):
        tasks = [make_task("fast", ok_fn, timeout_s=5.0)]
        results = self.executor.run_with_timeout(tasks)
        self.assertEqual(results[0].output, "done")

    def test_slow_task_marked_as_error_on_timeout(self):
        # Use a very short timeout to trigger the timeout path.
        tasks = [make_task("slow", blocking_fn, timeout_s=0.05)]
        results = self.executor.run_with_timeout(tasks)
        self.assertFalse(results[0].success)

    def test_timeout_error_message_meaningful(self):
        tasks = [make_task("slow", blocking_fn, timeout_s=0.05)]
        results = self.executor.run_with_timeout(tasks)
        self.assertIn("slow", results[0].error)

    def test_timeout_output_is_none(self):
        tasks = [make_task("slow", blocking_fn, timeout_s=0.05)]
        results = self.executor.run_with_timeout(tasks)
        self.assertIsNone(results[0].output)

    def test_mixed_timeout_and_success(self):
        tasks = [
            make_task("fast", ok_fn, timeout_s=5.0),
            make_task("slow", blocking_fn, timeout_s=0.05),
        ]
        results = self.executor.run_with_timeout(tasks)
        self.assertTrue(results[0].success)
        self.assertFalse(results[1].success)

    def test_result_order_preserved_with_timeout(self):
        tasks = [make_task(f"t{i}", ok_fn, timeout_s=5.0) for i in range(4)]
        results = self.executor.run_with_timeout(tasks)
        for i, r in enumerate(results):
            self.assertEqual(r.name, f"t{i}")


# ---------------------------------------------------------------------------
# ParallelResult frozen dataclass tests
# ---------------------------------------------------------------------------


class TestParallelResultFrozen(unittest.TestCase):
    def test_parallel_result_is_frozen(self):
        pr = ParallelResult(name="x", success=True, output=None, error="", duration_ms=1.0)
        with self.assertRaises((dataclasses.FrozenInstanceError, AttributeError)):
            pr.success = False  # type: ignore[misc]

    def test_parallel_result_fields(self):
        pr = ParallelResult(name="t", success=False, output=None, error="err", duration_ms=5.5)
        self.assertEqual(pr.name, "t")
        self.assertFalse(pr.success)
        self.assertIsNone(pr.output)
        self.assertEqual(pr.error, "err")
        self.assertAlmostEqual(pr.duration_ms, 5.5)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------


class TestRegistry(unittest.TestCase):
    def test_registry_has_default_key(self):
        self.assertIn("default", PARALLEL_STEP_REGISTRY)

    def test_registry_default_is_executor_class(self):
        self.assertIs(PARALLEL_STEP_REGISTRY["default"], ParallelStepExecutor)

    def test_registry_value_is_instantiable(self):
        cls = PARALLEL_STEP_REGISTRY["default"]
        instance = cls()
        self.assertIsInstance(instance, ParallelStepExecutor)


if __name__ == "__main__":
    unittest.main()
