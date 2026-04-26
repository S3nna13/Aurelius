"""Automatic model documentation from architecture inspection."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch.nn as nn


@dataclass
class ParameterBreakdown:
    """Breakdown of model parameters by component type."""

    total: int
    embed: int
    attention: int
    ffn: int
    lm_head: int
    other: int

    def summary(self) -> str:
        """Return a human-readable summary of the parameter breakdown."""

        def fmt(n: int) -> str:
            return f"{n / 1e6:.1f}M"

        lines = [
            f"Total parameters:  {fmt(self.total)} ({self.total:,})",
            f"  Embedding:       {fmt(self.embed)} ({self.embed / self.total:.1%})",
            f"  Attention:       {fmt(self.attention)} ({self.attention / self.total:.1%})",
            f"  FFN:             {fmt(self.ffn)} ({self.ffn / self.total:.1%})",
            f"  LM Head:         {fmt(self.lm_head)} ({self.lm_head / self.total:.1%})",
            f"  Other:           {fmt(self.other)} ({self.other / self.total:.1%})",
        ]
        return "\n".join(lines)


@dataclass
class ModelCard:
    """Structured model documentation card."""

    model_name: str = "Aurelius"
    architecture: str = "Decoder-only Transformer"
    parameters: ParameterBreakdown | None = None
    config_dict: dict = field(default_factory=dict)
    benchmark_results: dict[str, float] = field(default_factory=dict)
    training_notes: list[str] = field(default_factory=list)

    def to_markdown(self) -> str:
        """Generate a markdown model card."""
        lines = [
            f"# {self.model_name} Model Card",
            "",
            f"**Architecture:** {self.architecture}",
            "",
        ]

        if self.parameters:
            lines += ["## Parameters", "", "```", self.parameters.summary(), "```", ""]

        if self.config_dict:
            lines += ["## Configuration", "", "| Key | Value |", "|-----|-------|"]
            for k, v in self.config_dict.items():
                lines.append(f"| {k} | {v} |")
            lines.append("")

        if self.benchmark_results:
            lines += [
                "## Benchmark Results",
                "",
                "| Benchmark | Score |",
                "|-----------|-------|",
            ]
            for k, v in self.benchmark_results.items():
                lines.append(f"| {k} | {v:.4f} |")
            lines.append("")

        if self.training_notes:
            lines += ["## Training Notes", ""]
            for note in self.training_notes:
                lines.append(f"- {note}")
            lines.append("")

        return "\n".join(lines)


def count_parameters_by_component(model: nn.Module) -> ParameterBreakdown:
    """Break down parameter count by component type.

    Categories:
    - embed: parameters in 'embed' named modules
    - attention: parameters with 'attn' or 'q_proj'/'k_proj'/'v_proj'/'o_proj' in name
    - ffn: parameters with 'ffn' or 'gate_proj'/'up_proj'/'down_proj' in name
    - lm_head: parameters with 'lm_head' in name
    - other: everything else (norms, etc.)
    """
    counts = {"embed": 0, "attention": 0, "ffn": 0, "lm_head": 0, "other": 0}
    total = 0

    for name, param in model.named_parameters():
        n = param.numel()
        total += n

        if "embed" in name and "lm_head" not in name:
            counts["embed"] += n
        elif "lm_head" in name:
            counts["lm_head"] += n
        elif any(k in name for k in ("attn", "q_proj", "k_proj", "v_proj", "o_proj")):
            counts["attention"] += n
        elif any(k in name for k in ("ffn", "gate_proj", "up_proj", "down_proj")):
            counts["ffn"] += n
        else:
            counts["other"] += n

    return ParameterBreakdown(total=total, **counts)


def build_model_card(
    model: nn.Module,
    config: object | None = None,
    benchmark_results: dict[str, float] | None = None,
    model_name: str = "Aurelius",
) -> ModelCard:
    """Build a ModelCard from a model and optional config/benchmarks."""
    params = count_parameters_by_component(model)
    config_dict: dict = {}
    if config is not None:
        # Try to get __dict__ or dataclass fields
        if hasattr(config, "__dict__"):
            config_dict = {
                k: v
                for k, v in vars(config).items()
                if not k.startswith("_") and isinstance(v, (int, float, str, bool))
            }

    return ModelCard(
        model_name=model_name,
        parameters=params,
        config_dict=config_dict,
        benchmark_results=benchmark_results or {},
    )
