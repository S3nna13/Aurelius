"""HLM — Hybrid Language Model configuration and routing.

HLM defines a heterogeneous transformer where different layers use
different architectures: dense, MoE, MoD, ReMoDE, or Mamba SSM.

The layer_pattern specifies the sequence. For example::

    ["dense", "dense", "dense", "moe"] * 6

Creates a 24-layer model where every 4th layer is MoE and the rest are dense.

This is the architecture pattern used by Jamba (Hybrid Transformer-Mamba),
SAMBA, and other frontier hybrid models.

An HLMProfile captures the optimal architecture for a given task/constraint.
The HLMRouter selects the appropriate profile based on the task.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .config import HLM_PRESETS, HLMConfig, MoDConfig, MoEConfig, ReMoDEConfig


@dataclass
class HLMProfile:
    """A complete HLM architecture profile.

    Specifies the exact per-layer architecture for a hybrid model.
    """

    name: str
    n_layers: int
    d_model: int
    n_heads: int
    d_ff: int
    layer_pattern: list[str]
    moe_config: MoEConfig = field(default_factory=MoEConfig)
    mod_config: MoDConfig = field(default_factory=MoDConfig)
    remode_config: ReMoDEConfig = field(default_factory=ReMoDEConfig)
    total_params_estimate: int = 0
    flops_per_token_estimate: int = 0
    recommended_tasks: list[str] = field(default_factory=list)


# Pre-built profiles
HLM_PROFILES: dict[str, dict[str, Any]] = {
    "fast": {
        "name": "fast",
        "n_layers": 8,
        "d_model": 1024,
        "n_heads": 8,
        "d_ff": 4096,
        "layer_pattern": ["dense"] * 6 + ["moe"] * 2,
        "moe_config": MoEConfig(enabled=True, num_experts=4, top_k=1),
        "recommended_tasks": ["chat", "faq", "simple_qa"],
    },
    "balanced": {
        "name": "balanced",
        "n_layers": 24,
        "d_model": 2048,
        "n_heads": 16,
        "d_ff": 8192,
        "layer_pattern": ["dense"] * 18 + ["moe"] * 6,
        "moe_config": MoEConfig(enabled=True, num_experts=8, top_k=2),
        "recommended_tasks": ["general", "research", "analysis"],
    },
    "deepseek": {
        "name": "deepseek",
        "n_layers": 60,
        "d_model": 4096,
        "n_heads": 32,
        "d_ff": 16384,
        "layer_pattern": ["dense", "dense", "remode"] * 20,
        "remode_config": ReMoDEConfig(
            enabled=True, mod_capacity=0.3, moe_num_experts=128, moe_top_k=6
        ),
        "recommended_tasks": ["reasoning", "code", "math", "science"],
    },
    "mod_heavy": {
        "name": "mod_heavy",
        "n_layers": 24,
        "d_model": 2048,
        "n_heads": 16,
        "d_ff": 8192,
        "layer_pattern": ["mod"] * 12 + ["dense"] * 12,
        "mod_config": MoDConfig(enabled=True, capacity_factor=0.5),
        "recommended_tasks": ["long_context", "document_analysis"],
    },
    "efficient": {
        "name": "efficient",
        "n_layers": 16,
        "d_model": 1024,
        "n_heads": 8,
        "d_ff": 4096,
        "layer_pattern": ["mod", "dense", "moe", "dense"] * 4,
        "moe_config": MoEConfig(enabled=True, num_experts=4, top_k=1),
        "mod_config": MoDConfig(enabled=True, capacity_factor=0.5),
        "recommended_tasks": ["edge", "mobile", "low_latency"],
    },
}


class HLMRouter:
    """Routes tasks to the optimal HLM profile based on constraints.

    Considers:
    - Task type (chat, research, code, etc.)
    - Available VRAM / compute budget
    - Required context length
    - Latency SLA
    """

    def __init__(self, profiles: dict[str, dict[str, Any]] | None = None) -> None:
        self._profiles = profiles or HLM_PROFILES

    def select(
        self,
        task_type: str = "general",
        vram_gb: float = 16.0,
        latency_sla_ms: float = 5000.0,
        context_length: int = 8192,
    ) -> HLMProfile:
        """Select the best HLM profile given constraints."""

        # Low VRAM / edge deployment
        if vram_gb < 8 or latency_sla_ms < 1000:
            return self._build_profile("efficient")

        # Chat / simple tasks
        if task_type in ("chat", "faq", "simple_qa"):
            return self._build_profile("fast")

        # Long context tasks
        if context_length > 32768:
            return self._build_profile("mod_heavy")

        # Deep reasoning
        if task_type in ("reasoning", "code", "math", "science"):
            return self._build_profile("deepseek")

        # Default
        return self._build_profile("balanced")

    def list_profiles(self) -> list[str]:
        return list(self._profiles.keys())

    def _build_profile(self, name: str) -> HLMProfile:
        raw = self._profiles.get(name, self._profiles["balanced"])
        cfg = dict(raw)
        # Convert nested configs
        if "moe_config" in cfg and isinstance(cfg["moe_config"], dict):
            cfg["moe_config"] = MoEConfig(**cfg["moe_config"])
        if "mod_config" in cfg and isinstance(cfg["mod_config"], dict):
            cfg["mod_config"] = MoDConfig(**cfg["mod_config"])
        if "remode_config" in cfg and isinstance(cfg["remode_config"], dict):
            cfg["remode_config"] = ReMoDEConfig(**cfg["remode_config"])
        return HLMProfile(**cfg)


def get_hlm_config(profile_name: str = "balanced") -> HLMConfig:
    """Get an HLMConfig from a named profile."""
    raw = HLM_PRESETS.get(profile_name, HLM_PRESETS["hybrid_moe"])
    return HLMConfig(**raw)
