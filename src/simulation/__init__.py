"""Aurelius simulation surface: agent environment simulation and episode recording."""
__all__ = [
    "EnvState", "EnvAction", "EnvStep", "Environment", "GridWorldEnv", "ENV_REGISTRY",
    "AgentHarness", "AGENT_HARNESS",
    "Episode", "EpisodeRecorder", "EPISODE_RECORDER",
    "SIMULATION_REGISTRY",
]
from .environment import EnvState, EnvAction, EnvStep, Environment, GridWorldEnv, ENV_REGISTRY
from .agent_harness import AgentHarness, AGENT_HARNESS
from .episode_recorder import Episode, EpisodeRecorder, EPISODE_RECORDER

SIMULATION_REGISTRY: dict[str, object] = {
    "env_registry": ENV_REGISTRY,
    "harness": AGENT_HARNESS,
    "recorder": EPISODE_RECORDER,
}
