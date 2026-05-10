from __future__ import annotations

import torch
from aurelius.skills import SkillLibrary, SkillRegistry


def test_skill_registry_smoke():
    torch.manual_seed(0)
    registry = SkillRegistry(d_model=16, skill_dim=16, max_skills=16, n_top_k=2)
    hidden = torch.randn(1, 4, 16)

    output, skill = registry(hidden)

    assert output.shape == hidden.shape
    assert skill.shape[-1] == 16


def test_skill_library_smoke():
    torch.manual_seed(0)
    library = SkillLibrary(d_model=16, skill_dim=16, max_skills=16)
    hidden = torch.randn(1, 4, 16)

    output, skill = library(hidden)

    assert output.shape == hidden.shape
    assert skill.shape[-1] == 16
