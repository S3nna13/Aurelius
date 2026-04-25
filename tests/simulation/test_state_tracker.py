"""Tests for src/simulation/state_tracker.py (>=28 tests)."""

import unittest
from dataclasses import FrozenInstanceError

from src.simulation.state_tracker import (
    STATE_TRACKER_REGISTRY,
    REGISTRY,
    StateTracker,
    StateTransition,
)


class TestStateTransition(unittest.TestCase):
    def test_construction(self):
        st = StateTransition(
            from_state="a", action="go", to_state="b", reward=1.0, timestamp=0.0
        )
        self.assertEqual(st.from_state, "a")
        self.assertEqual(st.action, "go")
        self.assertEqual(st.to_state, "b")
        self.assertAlmostEqual(st.reward, 1.0)
        self.assertAlmostEqual(st.timestamp, 0.0)

    def test_frozen(self):
        st = StateTransition(from_state="a", action="x", to_state="b", reward=0.0, timestamp=0.0)
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            st.reward = 99.0  # type: ignore[misc]


class TestStateTrackerInitial(unittest.TestCase):
    def test_default_initial_state(self):
        tracker = StateTracker()
        self.assertEqual(tracker.current_state, "start")

    def test_custom_initial_state(self):
        tracker = StateTracker("idle")
        self.assertEqual(tracker.current_state, "idle")

    def test_empty_history(self):
        tracker = StateTracker()
        self.assertEqual(tracker.history(), [])

    def test_empty_action_counts(self):
        tracker = StateTracker()
        self.assertEqual(tracker.action_counts(), {})

    def test_total_reward_zero_initially(self):
        tracker = StateTracker()
        self.assertAlmostEqual(tracker.total_reward(), 0.0)


class TestStateTrackerTransition(unittest.TestCase):
    def setUp(self):
        self.tracker = StateTracker("s0")

    def test_transition_updates_current_state(self):
        self.tracker.transition("a", "s1")
        self.assertEqual(self.tracker.current_state, "s1")

    def test_transition_returns_state_transition(self):
        tr = self.tracker.transition("move", "s1", reward=2.5)
        self.assertIsInstance(tr, StateTransition)

    def test_transition_from_state(self):
        tr = self.tracker.transition("a", "s1")
        self.assertEqual(tr.from_state, "s0")

    def test_transition_to_state(self):
        tr = self.tracker.transition("a", "s1")
        self.assertEqual(tr.to_state, "s1")

    def test_transition_action(self):
        tr = self.tracker.transition("jump", "s1")
        self.assertEqual(tr.action, "jump")

    def test_transition_reward(self):
        tr = self.tracker.transition("a", "s1", reward=3.0)
        self.assertAlmostEqual(tr.reward, 3.0)

    def test_transition_timestamp_positive(self):
        tr = self.tracker.transition("a", "s1")
        self.assertGreater(tr.timestamp, 0.0)

    def test_transition_default_reward_zero(self):
        tr = self.tracker.transition("a", "s1")
        self.assertAlmostEqual(tr.reward, 0.0)


class TestStateTrackerHistory(unittest.TestCase):
    def setUp(self):
        self.tracker = StateTracker("s0")

    def test_history_grows_with_transitions(self):
        self.tracker.transition("a", "s1")
        self.tracker.transition("b", "s2")
        self.assertEqual(len(self.tracker.history()), 2)

    def test_history_returns_copy(self):
        self.tracker.transition("a", "s1")
        h = self.tracker.history()
        h.clear()
        self.assertEqual(len(self.tracker.history()), 1)

    def test_history_order(self):
        self.tracker.transition("a", "s1")
        self.tracker.transition("b", "s2")
        h = self.tracker.history()
        self.assertEqual(h[0].action, "a")
        self.assertEqual(h[1].action, "b")


class TestVisitedStates(unittest.TestCase):
    def test_includes_initial_when_no_transitions(self):
        tracker = StateTracker("home")
        self.assertIn("home", tracker.visited_states())

    def test_includes_from_and_to(self):
        tracker = StateTracker("s0")
        tracker.transition("a", "s1")
        vs = tracker.visited_states()
        self.assertIn("s0", vs)
        self.assertIn("s1", vs)

    def test_multiple_transitions(self):
        tracker = StateTracker("s0")
        tracker.transition("a", "s1")
        tracker.transition("b", "s2")
        vs = tracker.visited_states()
        self.assertGreaterEqual(len(vs), 3)


class TestActionCounts(unittest.TestCase):
    def test_counts_each_action(self):
        tracker = StateTracker()
        tracker.transition("run", "s1")
        tracker.transition("run", "s2")
        tracker.transition("jump", "s3")
        counts = tracker.action_counts()
        self.assertEqual(counts["run"], 2)
        self.assertEqual(counts["jump"], 1)

    def test_single_action(self):
        tracker = StateTracker()
        tracker.transition("x", "s1")
        self.assertEqual(tracker.action_counts(), {"x": 1})


class TestTotalReward(unittest.TestCase):
    def test_sums_rewards(self):
        tracker = StateTracker()
        tracker.transition("a", "s1", reward=1.0)
        tracker.transition("b", "s2", reward=2.5)
        self.assertAlmostEqual(tracker.total_reward(), 3.5)

    def test_negative_rewards(self):
        tracker = StateTracker()
        tracker.transition("a", "s1", reward=-1.0)
        tracker.transition("b", "s2", reward=3.0)
        self.assertAlmostEqual(tracker.total_reward(), 2.0)


class TestReset(unittest.TestCase):
    def test_reset_clears_history(self):
        tracker = StateTracker()
        tracker.transition("a", "s1")
        tracker.reset()
        self.assertEqual(tracker.history(), [])

    def test_reset_sets_state(self):
        tracker = StateTracker()
        tracker.transition("a", "s1")
        tracker.reset("home")
        self.assertEqual(tracker.current_state, "home")

    def test_reset_default_state(self):
        tracker = StateTracker("custom")
        tracker.reset()
        self.assertEqual(tracker.current_state, "start")

    def test_reset_clears_reward(self):
        tracker = StateTracker()
        tracker.transition("a", "s1", reward=10.0)
        tracker.reset()
        self.assertAlmostEqual(tracker.total_reward(), 0.0)


class TestToMarkovMatrix(unittest.TestCase):
    def _build(self):
        tracker = StateTracker("A")
        tracker.transition("go", "B")
        tracker.transition("go", "A")
        tracker.transition("stay", "A")
        return tracker

    def test_row_sums_to_one_for_visited(self):
        tracker = self._build()
        states = ["A", "B"]
        matrix = tracker.to_markov_matrix(states)
        for i, row in enumerate(matrix):
            row_sum = sum(row)
            # Rows for states that were departed from must sum to 1
            if row_sum > 0:
                self.assertAlmostEqual(row_sum, 1.0, places=10)

    def test_unvisited_row_is_zeros(self):
        tracker = StateTracker("A")
        tracker.transition("go", "B")
        states = ["A", "B", "C"]
        matrix = tracker.to_markov_matrix(states)
        # C is never departed from -> row all zeros
        c_idx = states.index("C")
        self.assertEqual(matrix[c_idx], [0.0, 0.0, 0.0])

    def test_deterministic_transition(self):
        tracker = StateTracker("X")
        tracker.transition("a", "Y")
        tracker.transition("a", "Y")
        states = ["X", "Y"]
        matrix = tracker.to_markov_matrix(states)
        # X always goes to Y
        self.assertAlmostEqual(matrix[0][1], 1.0)
        self.assertAlmostEqual(matrix[0][0], 0.0)

    def test_matrix_dimensions(self):
        tracker = StateTracker("A")
        tracker.transition("go", "B")
        states = ["A", "B", "C"]
        matrix = tracker.to_markov_matrix(states)
        self.assertEqual(len(matrix), 3)
        for row in matrix:
            self.assertEqual(len(row), 3)


class TestRegistry(unittest.TestCase):
    def test_has_default(self):
        self.assertIn("default", STATE_TRACKER_REGISTRY)

    def test_default_is_class(self):
        self.assertIs(STATE_TRACKER_REGISTRY["default"], StateTracker)

    def test_registry_alias(self):
        self.assertIs(REGISTRY, STATE_TRACKER_REGISTRY)

    def test_instantiable(self):
        cls = STATE_TRACKER_REGISTRY["default"]
        instance = cls()
        self.assertIsInstance(instance, StateTracker)


if __name__ == "__main__":
    unittest.main()
