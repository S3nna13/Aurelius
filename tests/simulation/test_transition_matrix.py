"""Tests for src/simulation/transition_matrix.py (>=28 tests)."""

import unittest

from src.simulation.transition_matrix import (
    REGISTRY,
    TRANSITION_MATRIX_REGISTRY,
    TransitionMatrix,
)


class TestTransitionMatrixBasics(unittest.TestCase):
    def test_state_index_found(self):
        tm = TransitionMatrix(["A", "B", "C"])
        self.assertEqual(tm.state_index("A"), 0)
        self.assertEqual(tm.state_index("B"), 1)
        self.assertEqual(tm.state_index("C"), 2)

    def test_state_index_not_found(self):
        tm = TransitionMatrix(["A", "B"])
        self.assertIsNone(tm.state_index("Z"))

    def test_empty_matrix_no_crash(self):
        tm = TransitionMatrix([])
        result = tm.normalize()
        self.assertEqual(result, [])

    def test_empty_stationary_distribution(self):
        tm = TransitionMatrix([])
        self.assertEqual(tm.stationary_distribution(), [])

    def test_single_state(self):
        tm = TransitionMatrix(["A"])
        self.assertIsNone(tm.state_index("B"))
        self.assertEqual(tm.state_index("A"), 0)


class TestAddTransition(unittest.TestCase):
    def test_add_transition_accumulates(self):
        tm = TransitionMatrix(["A", "B"])
        tm.add_transition("A", "B", weight=2.0)
        tm.add_transition("A", "B", weight=3.0)
        P = tm.normalize()
        # A -> B should be 1.0 (only path)
        self.assertAlmostEqual(P[0][1], 1.0)

    def test_add_transition_default_weight(self):
        tm = TransitionMatrix(["A", "B"])
        tm.add_transition("A", "B")  # weight=1.0
        P = tm.normalize()
        self.assertAlmostEqual(P[0][1], 1.0)

    def test_add_transition_unknown_state_raises(self):
        tm = TransitionMatrix(["A", "B"])
        with self.assertRaises(KeyError):
            tm.add_transition("A", "Z")

    def test_add_self_loop(self):
        tm = TransitionMatrix(["A", "B"])
        tm.add_transition("A", "A", weight=1.0)
        tm.add_transition("A", "B", weight=1.0)
        P = tm.normalize()
        self.assertAlmostEqual(P[0][0], 0.5)
        self.assertAlmostEqual(P[0][1], 0.5)


class TestNormalize(unittest.TestCase):
    def test_row_sums_to_one(self):
        tm = TransitionMatrix(["A", "B", "C"])
        tm.add_transition("A", "B", 1.0)
        tm.add_transition("A", "C", 3.0)
        P = tm.normalize()
        self.assertAlmostEqual(sum(P[0]), 1.0)

    def test_zero_row_stays_zero(self):
        tm = TransitionMatrix(["A", "B", "C"])
        tm.add_transition("A", "B")
        P = tm.normalize()
        # Row for B and C have no outgoing transitions
        self.assertEqual(P[1], [0.0, 0.0, 0.0])
        self.assertEqual(P[2], [0.0, 0.0, 0.0])

    def test_normalize_probabilities_correct(self):
        tm = TransitionMatrix(["A", "B"])
        tm.add_transition("A", "A", 1.0)
        tm.add_transition("A", "B", 3.0)
        P = tm.normalize()
        self.assertAlmostEqual(P[0][0], 0.25)
        self.assertAlmostEqual(P[0][1], 0.75)

    def test_all_rows_normalized(self):
        tm = TransitionMatrix(["A", "B", "C"])
        tm.add_transition("A", "B", 2.0)
        tm.add_transition("B", "C", 5.0)
        tm.add_transition("C", "A", 1.0)
        P = tm.normalize()
        for row in P:
            row_sum = sum(row)
            if row_sum > 0:
                self.assertAlmostEqual(row_sum, 1.0)


class TestStationaryDistribution(unittest.TestCase):
    def test_sums_to_one(self):
        tm = TransitionMatrix(["A", "B", "C"])
        tm.add_transition("A", "B")
        tm.add_transition("B", "C")
        tm.add_transition("C", "A")
        dist = tm.stationary_distribution()
        self.assertAlmostEqual(sum(dist), 1.0, places=6)

    def test_uniform_chain_uniform_stationary(self):
        # A<->B<->C in a symmetric cycle: stationary should be [1/3,1/3,1/3]
        tm = TransitionMatrix(["A", "B", "C"])
        tm.add_transition("A", "B")
        tm.add_transition("B", "C")
        tm.add_transition("C", "A")
        dist = tm.stationary_distribution()
        for v in dist:
            self.assertAlmostEqual(v, 1.0 / 3.0, places=5)

    def test_single_absorbing_state_stationary(self):
        tm = TransitionMatrix(["A", "B"])
        # B is absorbing; A -> B only
        tm.add_transition("A", "B")
        tm.add_transition("B", "B")
        dist = tm.stationary_distribution()
        self.assertAlmostEqual(sum(dist), 1.0, places=6)
        # Mass should converge to B
        self.assertGreater(dist[1], dist[0])

    def test_stationary_all_nonneg(self):
        tm = TransitionMatrix(["X", "Y"])
        tm.add_transition("X", "Y")
        tm.add_transition("Y", "X")
        dist = tm.stationary_distribution()
        for v in dist:
            self.assertGreaterEqual(v, 0.0)


class TestIsAbsorbing(unittest.TestCase):
    def test_absorbing_self_loop(self):
        tm = TransitionMatrix(["A", "B"])
        tm.add_transition("A", "A")
        self.assertTrue(tm.is_absorbing("A"))

    def test_not_absorbing_has_outgoing(self):
        tm = TransitionMatrix(["A", "B"])
        tm.add_transition("A", "B")
        self.assertFalse(tm.is_absorbing("A"))

    def test_not_absorbing_mixed(self):
        tm = TransitionMatrix(["A", "B"])
        tm.add_transition("A", "A", 1.0)
        tm.add_transition("A", "B", 1.0)
        self.assertFalse(tm.is_absorbing("A"))

    def test_unknown_state_raises(self):
        tm = TransitionMatrix(["A"])
        with self.assertRaises(KeyError):
            tm.is_absorbing("Z")

    def test_zero_row_not_absorbing(self):
        # No transitions at all -> normalized row all zeros -> diagonal is 0, not 1
        tm = TransitionMatrix(["A", "B"])
        self.assertFalse(tm.is_absorbing("A"))


class TestReachableFrom(unittest.TestCase):
    def test_direct_reachability(self):
        tm = TransitionMatrix(["A", "B", "C"])
        tm.add_transition("A", "B")
        reachable = tm.reachable_from("A")
        self.assertIn("A", reachable)
        self.assertIn("B", reachable)
        self.assertNotIn("C", reachable)

    def test_transitive_reachability(self):
        tm = TransitionMatrix(["A", "B", "C"])
        tm.add_transition("A", "B")
        tm.add_transition("B", "C")
        reachable = tm.reachable_from("A")
        self.assertIn("C", reachable)

    def test_isolated_state_reaches_only_itself(self):
        tm = TransitionMatrix(["A", "B"])
        # No transitions added
        reachable = tm.reachable_from("A")
        self.assertEqual(reachable, {"A"})

    def test_cycle_reachability(self):
        tm = TransitionMatrix(["A", "B", "C"])
        tm.add_transition("A", "B")
        tm.add_transition("B", "C")
        tm.add_transition("C", "A")
        reachable = tm.reachable_from("A")
        self.assertEqual(reachable, {"A", "B", "C"})

    def test_unknown_state_raises(self):
        tm = TransitionMatrix(["A"])
        with self.assertRaises(KeyError):
            tm.reachable_from("Z")


class TestRegistry(unittest.TestCase):
    def test_has_default(self):
        self.assertIn("default", TRANSITION_MATRIX_REGISTRY)

    def test_default_is_class(self):
        self.assertIs(TRANSITION_MATRIX_REGISTRY["default"], TransitionMatrix)

    def test_registry_alias(self):
        self.assertIs(REGISTRY, TRANSITION_MATRIX_REGISTRY)

    def test_instantiable(self):
        cls = TRANSITION_MATRIX_REGISTRY["default"]
        instance = cls(["A", "B"])
        self.assertIsInstance(instance, TransitionMatrix)


if __name__ == "__main__":
    unittest.main()
