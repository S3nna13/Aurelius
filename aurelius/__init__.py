"""Aurelius — Legacy compatibility namespace.

Provides the ``aurelius.*`` import path that some tests and legacy code
still reference.  Each top-level sub-package alias is eagerly registered
in ``sys.modules`` so that ``from aurelius.training.X import Y`` resolves
to ``src.training.X`` transparently.

Usage
-----
    from aurelius.training.gradient_surgery import GradientSurgeon
    from aurelius.alignment.constitutional_ai import ConstitutionalAI
"""

from __future__ import annotations

import importlib
import sys


def _alias(alias: str, target: str) -> None:
    """Register *alias* in ``sys.modules`` pointing to *target*."""
    mod = importlib.import_module(target)
    sys.modules.setdefault(alias, mod)


# ── Sub-package aliases (eager — breaks circular imports) ─────────
# Each call does importlib.import_module("src.X") and registers it
# as sys.modules["aurelius.X"].  From that point on Python resolves
# "aurelius.X.Y" as "src.X.Y" automatically.
_alias("aurelius.data", "src.data")
_alias("aurelius.eval", "src.eval")
_alias("aurelius.evaluation", "src.evaluation")
_alias("aurelius.training", "src.training")
_alias("aurelius.alignment", "src.alignment")
_alias("aurelius.inference", "src.inference")
_alias("aurelius.serving", "src.serving")
_alias("aurelius.safety", "src.safety")
_alias("aurelius.model", "src.model")
_alias("aurelius.agent", "src.agent")
_alias("aurelius.security", "src.security")
_alias("aurelius.memory", "src.memory")
_alias("aurelius.quantization", "src.quantization")
_alias("aurelius.routing", "src.routing")
_alias("aurelius.workflow", "src.workflow")
_alias("aurelius.multiagent", "src.multiagent")
_alias("aurelius.tools", "src.tools")
_alias("aurelius.compression", "src.compression")
_alias("aurelius.optimizers", "src.optimizers")
_alias("aurelius.interpretability", "src.interpretability")
_alias("aurelius.deployment", "src.deployment")
_alias("aurelius.monitoring", "src.monitoring")
_alias("aurelius.persona", "src.persona")
_alias("aurelius.cache", "src.cache")

# ── Agent registry (legacy names) ────────────────────────────────
from .agent_registry import (  # noqa: E402
    AGENT_REGISTRY,
    AGENTS_BY_CATEGORY,
    ALL_AGENTS,
    CREATIVE_AGENT,
    DEVOPS_AGENT,
    RESEARCH_AGENT,
    TUTOR_AGENT,
)

# ── Plugin system ────────────────────────────────────────────────
from .plugin_system import (  # noqa: E402
    BUILTIN_PLUGINS,
    PLUGIN_MANAGER,
    PluginManager,
)

# ── Skills registry ──────────────────────────────────────────────
from .skills_registry import (  # noqa: E402
    ALL_SKILLS,
    SKILL_REGISTRY,
    SKILLS_BY_CATEGORY,
)

__all__ = [
    "ALL_AGENTS",
    "AGENT_REGISTRY",
    "AGENTS_BY_CATEGORY",
    "CREATIVE_AGENT",
    "DEVOPS_AGENT",
    "RESEARCH_AGENT",
    "TUTOR_AGENT",
    "ALL_SKILLS",
    "SKILL_REGISTRY",
    "SKILLS_BY_CATEGORY",
    "BUILTIN_PLUGINS",
    "PLUGIN_MANAGER",
    "PluginManager",
]
