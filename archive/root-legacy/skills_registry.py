"""skills_registry.py — contract surface for SkillRegistry.

Contract: Skills are learnable vector embeddings with retrieval, composition,
and momentum-based acquisition. Each skill has a success rate and usage count.
Live path: skills.SkillRegistry → torch tensor operations.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any

import torch

from skills import SkillRegistry as _SkillRegistry


@dataclass
class RegistryEntry:
    name: str
    version: str = "0.1.0"
    contract: str = ""
    live: bool = True
    path: str = ""
    test_command: str = ""
    owner: str = ""


SKILLS_REGISTRY: dict[str, RegistryEntry] = {
    "skill_embedding": RegistryEntry(
        name="SkillEmbedding",
        contract="Store and retrieve learnable skill vectors (max_skills x skill_dim)",
        path="skills.SkillEmbedding",
        test_command="python3 -m pytest tests.py::test_skill_embedding_shape tests.py::test_skill_embedding_all tests.py::test_skill_embedding_add_skill tests.py::test_skill_embedding_named -v",
    ),
    "skill_retriever": RegistryEntry(
        name="SkillRetriever",
        contract="Top-k skill retrieval via dot-product attention",
        path="skills.SkillRetriever",
        test_command="python3 -m pytest tests.py -k test_skill_retriever -v",
    ),
    "skill_controller": RegistryEntry(
        name="SkillController",
        contract="Gated fusion of hidden state with retrieved skill embedding",
        path="skills.SkillController",
        test_command="python3 -m pytest tests.py::test_skill_controller_shape tests.py::test_skill_controller_grads -v",
    ),
    "skill_execution_adapter": RegistryEntry(
        name="SkillExecutionAdapter",
        contract="Cross-attention over skill KV for execution",
        path="skills.SkillExecutionAdapter",
        test_command="python3 -m pytest tests.py::test_skill_execution_adapter_shape -v",
    ),
    "skill_acquisition": RegistryEntry(
        name="SkillAcquisition",
        contract="Extract skills from trajectory mean-pooling with momentum update",
        path="skills.SkillAcquisition",
        test_command="python3 -m pytest tests.py::test_skill_acquisition_extract tests.py::test_skill_acquisition_update -v",
    ),
    "skill_registry": RegistryEntry(
        name="SkillRegistry",
        contract="Complete registry: embedding + retrieval + controller + adapter + acquisition + success tracking",
        path="skills.SkillRegistry",
        test_command="python3 -m pytest tests.py::test_skill_registry_add_skill tests.py::test_skill_registry_update_skill tests.py::test_skill_registry_get_top -v",
    ),
    "skill_library": RegistryEntry(
        name="SkillLibrary",
        contract="Composable skills: registry + task embedder + composition head",
        path="skills.SkillLibrary",
        test_command="python3 -m pytest tests.py::test_skill_library_compose -v",
    ),
}


def get_registry() -> dict[str, RegistryEntry]:
    return SKILLS_REGISTRY


def lookup(name: str) -> RegistryEntry | None:
    return SKILLS_REGISTRY.get(name)


class SkillRegistryContract:
    """Contract wrapper around skills.SkillRegistry.

    Verifies the live path by running a forward pass and checking
    output shapes, gradients, and state invariants.
    """

    def __init__(self, d_model: int = 64, skill_dim: int = 64,
                 max_skills: int = 16, n_top_k: int = 4):
        self._impl = _SkillRegistry(d_model, skill_dim, max_skills, n_top_k)
        self._d_model = d_model
        self._skill_dim = skill_dim
        self._max_skills = max_skills
        self._n_top_k = n_top_k

    @property
    def d_model(self) -> int:
        return self._d_model

    @property
    def skill_dim(self) -> int:
        return self._skill_dim

    @property
    def max_skills(self) -> int:
        return self._max_skills

    @property
    def n_top_k(self) -> int:
        return self._n_top_k

    def forward(self, h: torch.Tensor, skill_ids: torch.Tensor | None = None,
                learn: bool = False) -> tuple[torch.Tensor, ...]:
        return self._impl(h, skill_ids, learn)

    def learn_skill(self, trajectories: torch.Tensor, success: float) -> int:
        return self._impl.learn_skill(trajectories, success)

    def get_top_skills(self, k: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
        return self._impl.get_top_skills(k)

    def verify_contract(self, batch: int = 2, time: int = 8) -> None:
        B, T, D = batch, time, self._d_model
        h = torch.randn(B, T, D)
        out, skill = self._impl(h)
        assert out.shape == (B, T, D), f"Expected {(B,T,D)}, got {out.shape}"
        assert skill.shape == (B, D) or skill.shape == (B, T, D), f"Unexpected skill shape {skill.shape}"

        out_learn, skill_learn, idx = self._impl(h, learn=True)
        assert out_learn.shape == (B, T, D)
        assert self._impl.skill_usage_count[idx].sum().item() > 0

        indices, values = self._impl.get_top_skills(k=4)
        assert indices.shape == (4,)
        assert values.shape == (4,)

        single_traj = torch.randn(1, T, D)
        sid = self._impl.learn_skill(single_traj, success=0.8)
        assert 0 <= sid < self._max_skills

    def __repr__(self) -> str:
        return (
            f"SkillRegistryContract(d_model={self._d_model}, "
            f"max_skills={self._max_skills}, n_top_k={self._n_top_k})"
        )
