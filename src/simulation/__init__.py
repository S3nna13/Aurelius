"""Aurelius simulation surface: agent environment simulation and episode recording."""
__all__ = [
    "EnvState", "EnvAction", "EnvStep", "Environment", "GridWorldEnv", "ENV_REGISTRY",
    "AgentHarness", "AGENT_HARNESS",
    "Episode", "EpisodeRecorder", "EPISODE_RECORDER",
    "RewardShapeType", "RewardShaperConfig", "PotentialFunction", "RewardShaper",
    "AgentAction", "MultiAgentStep", "MultiAgentEnv", "SimpleCooperativeEnv",
    "CurriculumLevel", "CurriculumEnv",
    "SIMULATION_REGISTRY",
]
from .environment import EnvState, EnvAction, EnvStep, Environment, GridWorldEnv, ENV_REGISTRY
from .agent_harness import AgentHarness, AGENT_HARNESS
from .episode_recorder import Episode, EpisodeRecorder, EPISODE_RECORDER
from .reward_shaper import RewardShapeType, RewardShaperConfig, PotentialFunction, RewardShaper
from .multi_agent_env import AgentAction, MultiAgentStep, MultiAgentEnv, SimpleCooperativeEnv
from .curriculum_env import CurriculumLevel, CurriculumEnv

SIMULATION_REGISTRY: dict[str, object] = {
    "env_registry": ENV_REGISTRY,
    "harness": AGENT_HARNESS,
    "recorder": EPISODE_RECORDER,
    "reward_shaper": RewardShaper,
    "multi_agent_env": SimpleCooperativeEnv,
    "curriculum_env": CurriculumEnv,
}
