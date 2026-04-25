"""
Tests for src/computer_use/task_decomposer.py
"""

import unittest

from src.computer_use.task_decomposer import (
    AtomicStep,
    DecomposedTask,
    TaskDecomposer,
    TASK_DECOMPOSER_REGISTRY,
)


class TestAtomicStep(unittest.TestCase):
    def test_fields(self):
        step = AtomicStep(step_id=0, description="do something", action_type="CLICK", target="btn", estimated_ms=150)
        self.assertEqual(step.step_id, 0)
        self.assertEqual(step.description, "do something")
        self.assertEqual(step.action_type, "CLICK")
        self.assertEqual(step.target, "btn")
        self.assertEqual(step.estimated_ms, 150)

    def test_frozen(self):
        step = AtomicStep(step_id=0, description="d", action_type="WAIT")
        with self.assertRaises(Exception):
            step.step_id = 99  # type: ignore[misc]

    def test_defaults(self):
        step = AtomicStep(step_id=1, description="d", action_type="SCREENSHOT")
        self.assertEqual(step.target, "")
        self.assertEqual(step.estimated_ms, 100)


class TestDecomposedTask(unittest.TestCase):
    def test_fields(self):
        steps = [AtomicStep(step_id=0, description="s", action_type="SCREENSHOT")]
        task = DecomposedTask(task_description="check screen", steps=steps, total_estimated_ms=100)
        self.assertEqual(task.task_description, "check screen")
        self.assertEqual(len(task.steps), 1)
        self.assertEqual(task.total_estimated_ms, 100)

    def test_frozen(self):
        steps: list[AtomicStep] = []
        task = DecomposedTask(task_description="t", steps=steps, total_estimated_ms=0)
        with self.assertRaises(Exception):
            task.task_description = "modified"  # type: ignore[misc]


class TestTaskDecomposer(unittest.TestCase):
    def setUp(self):
        self.decomposer = TaskDecomposer()

    # --- "open" / "launch" ---
    def test_open_has_navigate_step(self):
        task = self.decomposer.decompose("open browser")
        action_types = [s.action_type for s in task.steps]
        self.assertIn("NAVIGATE", action_types)

    def test_launch_has_navigate_step(self):
        task = self.decomposer.decompose("launch the application")
        action_types = [s.action_type for s in task.steps]
        self.assertIn("NAVIGATE", action_types)

    def test_open_has_screenshot_step(self):
        task = self.decomposer.decompose("open browser")
        action_types = [s.action_type for s in task.steps]
        self.assertIn("SCREENSHOT", action_types)

    def test_open_two_steps(self):
        task = self.decomposer.decompose("open browser")
        self.assertEqual(len(task.steps), 2)

    # --- "type" / "enter" ---
    def test_type_has_click_step(self):
        task = self.decomposer.decompose("type hello in the search box")
        action_types = [s.action_type for s in task.steps]
        self.assertIn("CLICK", action_types)

    def test_type_has_type_step(self):
        task = self.decomposer.decompose("type hello in the search box")
        action_types = [s.action_type for s in task.steps]
        self.assertIn("TYPE", action_types)

    def test_type_has_screenshot_step(self):
        task = self.decomposer.decompose("type hello in the search box")
        action_types = [s.action_type for s in task.steps]
        self.assertIn("SCREENSHOT", action_types)

    def test_enter_has_type_step(self):
        task = self.decomposer.decompose("enter the password")
        action_types = [s.action_type for s in task.steps]
        self.assertIn("TYPE", action_types)

    def test_type_three_steps(self):
        task = self.decomposer.decompose("type hello")
        self.assertEqual(len(task.steps), 3)

    # --- "click" ---
    def test_click_has_click_step(self):
        task = self.decomposer.decompose("click the submit button")
        action_types = [s.action_type for s in task.steps]
        self.assertIn("CLICK", action_types)

    def test_click_has_screenshot_step(self):
        task = self.decomposer.decompose("click the submit button")
        action_types = [s.action_type for s in task.steps]
        self.assertIn("SCREENSHOT", action_types)

    def test_click_two_steps(self):
        task = self.decomposer.decompose("click button")
        self.assertEqual(len(task.steps), 2)

    # --- "scroll" ---
    def test_scroll_has_scroll_step(self):
        task = self.decomposer.decompose("scroll down the page")
        action_types = [s.action_type for s in task.steps]
        self.assertIn("SCROLL", action_types)

    def test_scroll_has_screenshot_step(self):
        task = self.decomposer.decompose("scroll down the page")
        action_types = [s.action_type for s in task.steps]
        self.assertIn("SCREENSHOT", action_types)

    def test_scroll_two_steps(self):
        task = self.decomposer.decompose("scroll down")
        self.assertEqual(len(task.steps), 2)

    # --- default (no keyword match) ---
    def test_default_has_screenshot_step(self):
        task = self.decomposer.decompose("check screen")
        action_types = [s.action_type for s in task.steps]
        self.assertIn("SCREENSHOT", action_types)

    def test_default_has_wait_step(self):
        task = self.decomposer.decompose("check screen")
        action_types = [s.action_type for s in task.steps]
        self.assertIn("WAIT", action_types)

    def test_default_two_steps(self):
        task = self.decomposer.decompose("do something unrecognized")
        self.assertEqual(len(task.steps), 2)

    def test_default_unrecognized_task(self):
        task = self.decomposer.decompose("analyze the report")
        action_types = [s.action_type for s in task.steps]
        self.assertIn("SCREENSHOT", action_types)
        self.assertIn("WAIT", action_types)

    # --- total_estimated_ms ---
    def test_total_estimated_ms_is_sum(self):
        task = self.decomposer.decompose("open browser")
        expected = sum(s.estimated_ms for s in task.steps)
        self.assertEqual(task.total_estimated_ms, expected)

    def test_total_estimated_ms_type_task(self):
        task = self.decomposer.decompose("type hello")
        expected = sum(s.estimated_ms for s in task.steps)
        self.assertEqual(task.total_estimated_ms, expected)

    def test_total_estimated_ms_positive(self):
        task = self.decomposer.decompose("anything random here")
        self.assertGreater(task.total_estimated_ms, 0)

    # --- step_id ordering ---
    def test_step_ids_sequential(self):
        task = self.decomposer.decompose("open app")
        for idx, step in enumerate(task.steps):
            self.assertEqual(step.step_id, idx)

    # --- to_actions ---
    def test_to_actions_returns_list_of_dicts(self):
        task = self.decomposer.decompose("open browser")
        actions = self.decomposer.to_actions(task)
        self.assertIsInstance(actions, list)
        for item in actions:
            self.assertIsInstance(item, dict)

    def test_to_actions_has_step_id(self):
        task = self.decomposer.decompose("click button")
        actions = self.decomposer.to_actions(task)
        for item in actions:
            self.assertIn("step_id", item)

    def test_to_actions_has_action_type(self):
        task = self.decomposer.decompose("click button")
        actions = self.decomposer.to_actions(task)
        for item in actions:
            self.assertIn("action_type", item)

    def test_to_actions_has_target(self):
        task = self.decomposer.decompose("click button")
        actions = self.decomposer.to_actions(task)
        for item in actions:
            self.assertIn("target", item)

    # --- REGISTRY ---
    def test_registry_exists(self):
        self.assertIn("default", TASK_DECOMPOSER_REGISTRY)

    def test_registry_default_is_class(self):
        self.assertIs(TASK_DECOMPOSER_REGISTRY["default"], TaskDecomposer)

    def test_registry_default_instantiable(self):
        cls = TASK_DECOMPOSER_REGISTRY["default"]
        instance = cls()
        self.assertIsInstance(instance, TaskDecomposer)


if __name__ == "__main__":
    unittest.main()
