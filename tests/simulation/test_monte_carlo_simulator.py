"""Tests for src/simulation/monte_carlo_simulator.py (>=28 tests)."""

import statistics
import unittest
from dataclasses import FrozenInstanceError

from src.simulation.monte_carlo_simulator import (
    MONTE_CARLO_REGISTRY,
    REGISTRY,
    MCConfig,
    MCEpisode,
    MonteCarloSimulator,
)


class TestMCConfig(unittest.TestCase):
    def test_defaults(self):
        cfg = MCConfig()
        self.assertEqual(cfg.num_episodes, 1000)
        self.assertAlmostEqual(cfg.gamma, 0.99)
        self.assertEqual(cfg.seed, 42)

    def test_custom_values(self):
        cfg = MCConfig(num_episodes=500, gamma=0.95, seed=7)
        self.assertEqual(cfg.num_episodes, 500)
        self.assertAlmostEqual(cfg.gamma, 0.95)
        self.assertEqual(cfg.seed, 7)

    def test_frozen(self):
        cfg = MCConfig()
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            cfg.num_episodes = 1  # type: ignore[misc]


class TestMCEpisode(unittest.TestCase):
    def test_construction(self):
        ep = MCEpisode(episode_id=1, rewards=[1.0], returns=[1.0], total_return=1.0)
        self.assertEqual(ep.episode_id, 1)
        self.assertEqual(ep.rewards, [1.0])
        self.assertEqual(ep.returns, [1.0])
        self.assertAlmostEqual(ep.total_return, 1.0)

    def test_frozen(self):
        ep = MCEpisode(episode_id=0, rewards=[], returns=[], total_return=0.0)
        with self.assertRaises((FrozenInstanceError, AttributeError)):
            ep.episode_id = 99  # type: ignore[misc]

    def test_empty_rewards(self):
        ep = MCEpisode(episode_id=0, rewards=[], returns=[], total_return=0.0)
        self.assertEqual(ep.rewards, [])
        self.assertEqual(ep.returns, [])


class TestComputeReturns(unittest.TestCase):
    def setUp(self):
        self.sim = MonteCarloSimulator()

    def test_single_step(self):
        result = self.sim.compute_returns([5.0])
        self.assertEqual(len(result), 1)
        self.assertAlmostEqual(result[0], 5.0)

    def test_two_steps_discounted(self):
        # G_0 = 1 + 0.99*2 = 2.98, G_1 = 2
        result = self.sim.compute_returns([1.0, 2.0])
        self.assertAlmostEqual(result[1], 2.0)
        self.assertAlmostEqual(result[0], 1.0 + 0.99 * 2.0)

    def test_multi_step_discounted(self):
        rewards = [1.0, 1.0, 1.0]
        result = self.sim.compute_returns(rewards, gamma=0.5)
        # G_2=1, G_1=1+0.5*1=1.5, G_0=1+0.5*1.5=1.75
        self.assertAlmostEqual(result[2], 1.0)
        self.assertAlmostEqual(result[1], 1.5)
        self.assertAlmostEqual(result[0], 1.75)

    def test_gamma_zero(self):
        result = self.sim.compute_returns([3.0, 7.0, 2.0], gamma=0.0)
        # With gamma=0, G_t = r_t
        self.assertAlmostEqual(result[0], 3.0)
        self.assertAlmostEqual(result[1], 7.0)
        self.assertAlmostEqual(result[2], 2.0)

    def test_gamma_one(self):
        result = self.sim.compute_returns([1.0, 1.0, 1.0], gamma=1.0)
        self.assertAlmostEqual(result[0], 3.0)
        self.assertAlmostEqual(result[1], 2.0)
        self.assertAlmostEqual(result[2], 1.0)

    def test_empty_rewards(self):
        result = self.sim.compute_returns([])
        self.assertEqual(result, [])

    def test_length_preserved(self):
        rewards = [0.5] * 10
        result = self.sim.compute_returns(rewards)
        self.assertEqual(len(result), 10)

    def test_uses_config_gamma_by_default(self):
        cfg = MCConfig(gamma=0.5)
        sim = MonteCarloSimulator(cfg)
        result = sim.compute_returns([2.0, 2.0])
        self.assertAlmostEqual(result[0], 2.0 + 0.5 * 2.0)

    def test_override_gamma(self):
        sim = MonteCarloSimulator(MCConfig(gamma=0.99))
        result = sim.compute_returns([1.0, 0.0], gamma=0.0)
        self.assertAlmostEqual(result[0], 1.0)


class TestRunEpisode(unittest.TestCase):
    def setUp(self):
        self.sim = MonteCarloSimulator()

    def test_calls_reward_fn_max_steps_times(self):
        call_log = []

        def reward_fn(step):
            call_log.append(step)
            return 1.0

        self.sim.run_episode(reward_fn, max_steps=10)
        self.assertEqual(len(call_log), 10)

    def test_steps_passed_in_order(self):
        steps_seen = []

        def reward_fn(step):
            steps_seen.append(step)
            return 0.0

        self.sim.run_episode(reward_fn, max_steps=5)
        self.assertEqual(steps_seen, [0, 1, 2, 3, 4])

    def test_returns_mc_episode(self):
        ep = self.sim.run_episode(lambda s: 1.0, max_steps=3)
        self.assertIsInstance(ep, MCEpisode)

    def test_total_return_equals_returns_zero(self):
        ep = self.sim.run_episode(lambda s: 1.0, max_steps=5)
        self.assertAlmostEqual(ep.total_return, ep.returns[0])

    def test_rewards_length_equals_max_steps(self):
        ep = self.sim.run_episode(lambda s: 2.0, max_steps=7)
        self.assertEqual(len(ep.rewards), 7)

    def test_zero_reward_fn(self):
        ep = self.sim.run_episode(lambda s: 0.0, max_steps=10)
        self.assertAlmostEqual(ep.total_return, 0.0)

    def test_default_max_steps(self):
        ep = self.sim.run_episode(lambda s: 1.0)
        self.assertEqual(len(ep.rewards), 200)


class TestEstimateValue(unittest.TestCase):
    def setUp(self):
        self.sim = MonteCarloSimulator()

    def _make_episode(self, total):
        return MCEpisode(episode_id=0, rewards=[], returns=[], total_return=total)

    def test_single_episode(self):
        ep = self._make_episode(5.0)
        self.assertAlmostEqual(self.sim.estimate_value([ep]), 5.0)

    def test_multiple_episodes_mean(self):
        episodes = [self._make_episode(v) for v in [2.0, 4.0, 6.0]]
        self.assertAlmostEqual(self.sim.estimate_value(episodes), 4.0)

    def test_empty_list(self):
        self.assertAlmostEqual(self.sim.estimate_value([]), 0.0)


class TestValueVariance(unittest.TestCase):
    def setUp(self):
        self.sim = MonteCarloSimulator()

    def _make_episode(self, total):
        return MCEpisode(episode_id=0, rewards=[], returns=[], total_return=total)

    def test_single_episode_returns_zero(self):
        ep = self._make_episode(3.0)
        self.assertAlmostEqual(self.sim.value_variance([ep]), 0.0)

    def test_empty_returns_zero(self):
        self.assertAlmostEqual(self.sim.value_variance([]), 0.0)

    def test_two_identical_episodes_zero_variance(self):
        episodes = [self._make_episode(2.0), self._make_episode(2.0)]
        self.assertAlmostEqual(self.sim.value_variance(episodes), 0.0)

    def test_pstdev_matches_statistics(self):
        values = [1.0, 2.0, 3.0, 4.0, 5.0]
        episodes = [self._make_episode(v) for v in values]
        expected = statistics.pstdev(values)
        self.assertAlmostEqual(self.sim.value_variance(episodes), expected, places=10)


class TestRegistry(unittest.TestCase):
    def test_registry_has_default(self):
        self.assertIn("default", MONTE_CARLO_REGISTRY)

    def test_registry_default_is_class(self):
        self.assertIs(MONTE_CARLO_REGISTRY["default"], MonteCarloSimulator)

    def test_registry_alias(self):
        self.assertIs(REGISTRY, MONTE_CARLO_REGISTRY)

    def test_registry_instantiable(self):
        cls = MONTE_CARLO_REGISTRY["default"]
        instance = cls()
        self.assertIsInstance(instance, MonteCarloSimulator)


if __name__ == "__main__":
    unittest.main()
