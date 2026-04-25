"""
Tests for src/workflow/retry_workflow.py
≥28 test cases covering RetryStrategy, StepResult, RetryWorkflow, and REGISTRY.
"""

import dataclasses
import unittest

from src.workflow.retry_workflow import (
    RETRY_WORKFLOW_REGISTRY,
    RetryStrategy,
    RetryWorkflow,
    StepResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def always_succeed():
    return "ok"


def always_fail():
    raise ValueError("boom")


def succeed_after(n: int):
    """Returns a function that fails n times then succeeds."""
    call_count = [0]

    def fn():
        call_count[0] += 1
        if call_count[0] <= n:
            raise RuntimeError(f"fail #{call_count[0]}")
        return "success"

    return fn


# ---------------------------------------------------------------------------
# delay_for_attempt tests
# ---------------------------------------------------------------------------

class TestDelayForAttempt(unittest.TestCase):

    def test_immediate_always_zero(self):
        rw = RetryWorkflow(strategy=RetryStrategy.IMMEDIATE, base_delay_s=5.0)
        for attempt in range(5):
            self.assertEqual(rw.delay_for_attempt(attempt), 0.0)

    def test_fixed_delay_always_base(self):
        rw = RetryWorkflow(strategy=RetryStrategy.FIXED_DELAY, base_delay_s=2.5)
        for attempt in range(5):
            self.assertEqual(rw.delay_for_attempt(attempt), 2.5)

    def test_exponential_backoff_attempt_0(self):
        rw = RetryWorkflow(strategy=RetryStrategy.EXPONENTIAL_BACKOFF, base_delay_s=1.0)
        self.assertEqual(rw.delay_for_attempt(0), 1.0)

    def test_exponential_backoff_attempt_1(self):
        rw = RetryWorkflow(strategy=RetryStrategy.EXPONENTIAL_BACKOFF, base_delay_s=1.0)
        self.assertEqual(rw.delay_for_attempt(1), 2.0)

    def test_exponential_backoff_attempt_2(self):
        rw = RetryWorkflow(strategy=RetryStrategy.EXPONENTIAL_BACKOFF, base_delay_s=1.0)
        self.assertEqual(rw.delay_for_attempt(2), 4.0)

    def test_exponential_backoff_attempt_3(self):
        rw = RetryWorkflow(strategy=RetryStrategy.EXPONENTIAL_BACKOFF, base_delay_s=1.0)
        self.assertEqual(rw.delay_for_attempt(3), 8.0)

    def test_exponential_backoff_scales_with_base(self):
        rw = RetryWorkflow(strategy=RetryStrategy.EXPONENTIAL_BACKOFF, base_delay_s=0.5)
        self.assertAlmostEqual(rw.delay_for_attempt(2), 2.0)

    def test_fixed_delay_different_base(self):
        rw = RetryWorkflow(strategy=RetryStrategy.FIXED_DELAY, base_delay_s=7.0)
        self.assertEqual(rw.delay_for_attempt(0), 7.0)
        self.assertEqual(rw.delay_for_attempt(10), 7.0)


# ---------------------------------------------------------------------------
# run_step tests
# ---------------------------------------------------------------------------

class TestRunStep(unittest.TestCase):

    def _noop_rw(self, max_retries=3):
        return RetryWorkflow(max_retries=max_retries, sleep_fn=lambda x: None)

    def test_success_first_try(self):
        rw = self._noop_rw()
        result = rw.run_step("step1", always_succeed)
        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 1)
        self.assertEqual(result.output, "ok")
        self.assertEqual(result.error, "")

    def test_success_after_retries(self):
        rw = self._noop_rw(max_retries=3)
        fn = succeed_after(2)
        result = rw.run_step("step2", fn)
        self.assertTrue(result.success)
        self.assertEqual(result.attempts, 3)

    def test_all_fail_returns_failure(self):
        rw = self._noop_rw(max_retries=2)
        result = rw.run_step("step3", always_fail)
        self.assertFalse(result.success)

    def test_all_fail_attempts_equals_max_retries_plus_one(self):
        rw = self._noop_rw(max_retries=2)
        result = rw.run_step("step4", always_fail)
        self.assertEqual(result.attempts, 3)

    def test_all_fail_error_message_captured(self):
        rw = self._noop_rw(max_retries=1)
        result = rw.run_step("step5", always_fail)
        self.assertIn("boom", result.error)

    def test_step_name_in_result(self):
        rw = self._noop_rw()
        result = rw.run_step("my_step", always_succeed)
        self.assertEqual(result.step_name, "my_step")

    def test_output_none_on_failure(self):
        rw = self._noop_rw(max_retries=0)
        result = rw.run_step("step6", always_fail)
        self.assertIsNone(result.output)

    def test_sleep_fn_called_between_retries(self):
        calls = []
        rw = RetryWorkflow(
            max_retries=3,
            strategy=RetryStrategy.FIXED_DELAY,
            base_delay_s=1.0,
            sleep_fn=calls.append,
        )
        rw.run_step("s", always_fail)
        # Should be called max_retries times (not after the last attempt)
        self.assertEqual(len(calls), 3)

    def test_sleep_fn_not_called_on_success(self):
        calls = []
        rw = RetryWorkflow(max_retries=3, sleep_fn=calls.append)
        rw.run_step("s", always_succeed)
        self.assertEqual(len(calls), 0)

    def test_sleep_fn_delay_values_immediate(self):
        calls = []
        rw = RetryWorkflow(
            max_retries=2,
            strategy=RetryStrategy.IMMEDIATE,
            base_delay_s=5.0,
            sleep_fn=calls.append,
        )
        rw.run_step("s", always_fail)
        self.assertTrue(all(v == 0.0 for v in calls))

    def test_sleep_fn_delay_values_fixed(self):
        calls = []
        rw = RetryWorkflow(
            max_retries=3,
            strategy=RetryStrategy.FIXED_DELAY,
            base_delay_s=2.0,
            sleep_fn=calls.append,
        )
        rw.run_step("s", always_fail)
        self.assertTrue(all(v == 2.0 for v in calls))

    def test_max_retries_zero_only_one_attempt(self):
        rw = RetryWorkflow(max_retries=0, sleep_fn=lambda x: None)
        result = rw.run_step("s", always_fail)
        self.assertEqual(result.attempts, 1)


# ---------------------------------------------------------------------------
# StepResult frozen dataclass tests
# ---------------------------------------------------------------------------

class TestStepResultFrozen(unittest.TestCase):

    def test_step_result_is_frozen(self):
        sr = StepResult(step_name="x", success=True, attempts=1, output=None)
        with self.assertRaises((dataclasses.FrozenInstanceError, AttributeError)):
            sr.success = False  # type: ignore[misc]

    def test_step_result_default_error_empty(self):
        sr = StepResult(step_name="x", success=True, attempts=1, output=42)
        self.assertEqual(sr.error, "")

    def test_step_result_fields(self):
        sr = StepResult(step_name="abc", success=False, attempts=5, output=None, error="err")
        self.assertEqual(sr.step_name, "abc")
        self.assertFalse(sr.success)
        self.assertEqual(sr.attempts, 5)
        self.assertIsNone(sr.output)
        self.assertEqual(sr.error, "err")


# ---------------------------------------------------------------------------
# run_pipeline tests
# ---------------------------------------------------------------------------

class TestRunPipeline(unittest.TestCase):

    def _noop_rw(self):
        return RetryWorkflow(max_retries=0, sleep_fn=lambda x: None)

    def test_all_success_returns_all_results(self):
        rw = self._noop_rw()
        steps = [("a", lambda: 1), ("b", lambda: 2), ("c", lambda: 3)]
        results = rw.run_pipeline(steps)
        self.assertEqual(len(results), 3)
        self.assertTrue(all(r.success for r in results))

    def test_stops_on_first_failure(self):
        rw = self._noop_rw()
        steps = [("a", lambda: 1), ("b", always_fail), ("c", lambda: 3)]
        results = rw.run_pipeline(steps)
        # Should have results for "a" and "b" only
        self.assertEqual(len(results), 2)
        self.assertTrue(results[0].success)
        self.assertFalse(results[1].success)

    def test_empty_pipeline_returns_empty_list(self):
        rw = self._noop_rw()
        results = rw.run_pipeline([])
        self.assertEqual(results, [])

    def test_pipeline_result_order_preserved(self):
        rw = self._noop_rw()
        steps = [("first", lambda: "A"), ("second", lambda: "B")]
        results = rw.run_pipeline(steps)
        self.assertEqual(results[0].step_name, "first")
        self.assertEqual(results[1].step_name, "second")

    def test_pipeline_outputs_captured(self):
        rw = self._noop_rw()
        steps = [("x", lambda: 99)]
        results = rw.run_pipeline(steps)
        self.assertEqual(results[0].output, 99)

    def test_single_failing_step(self):
        rw = self._noop_rw()
        results = rw.run_pipeline([("only", always_fail)])
        self.assertEqual(len(results), 1)
        self.assertFalse(results[0].success)


# ---------------------------------------------------------------------------
# Registry tests
# ---------------------------------------------------------------------------

class TestRegistry(unittest.TestCase):

    def test_registry_has_default_key(self):
        self.assertIn("default", RETRY_WORKFLOW_REGISTRY)

    def test_registry_default_is_retry_workflow_class(self):
        self.assertIs(RETRY_WORKFLOW_REGISTRY["default"], RetryWorkflow)

    def test_registry_value_is_instantiable(self):
        cls = RETRY_WORKFLOW_REGISTRY["default"]
        instance = cls()
        self.assertIsInstance(instance, RetryWorkflow)


if __name__ == "__main__":
    unittest.main()
