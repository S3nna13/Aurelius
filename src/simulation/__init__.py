"""Aurelius simulation surface: agent environment simulation and episode recording."""

__all__ = [
    "EnvState",
    "EnvAction",
    "EnvStep",
    "Environment",
    "GridWorldEnv",
    "ENV_REGISTRY",
    "AgentHarness",
    "AGENT_HARNESS",
    "Episode",
    "EpisodeRecorder",
    "EPISODE_RECORDER",
    "RewardShapeType",
    "RewardShaperConfig",
    "PotentialFunction",
    "RewardShaper",
    "AgentAction",
    "MultiAgentStep",
    "MultiAgentEnv",
    "SimpleCooperativeEnv",
    "CurriculumLevel",
    "CurriculumEnv",
    "SIMULATION_REGISTRY",
]
from .agent_harness import AGENT_HARNESS, AgentHarness
from .curriculum_env import CurriculumEnv, CurriculumLevel
from .environment import ENV_REGISTRY, EnvAction, Environment, EnvState, EnvStep, GridWorldEnv
from .episode_recorder import EPISODE_RECORDER, Episode, EpisodeRecorder
from .multi_agent_env import AgentAction, MultiAgentEnv, MultiAgentStep, SimpleCooperativeEnv
from .reward_shaper import PotentialFunction, RewardShaper, RewardShaperConfig, RewardShapeType

SIMULATION_REGISTRY: dict[str, object] = {
    "env_registry": ENV_REGISTRY,
    "harness": AGENT_HARNESS,
    "recorder": EPISODE_RECORDER,
    "reward_shaper": RewardShaper,
    "multi_agent_env": SimpleCooperativeEnv,
    "curriculum_env": CurriculumEnv,
}
