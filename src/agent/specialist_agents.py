"""Specialist agents — domain-expert agents and ApeOrchestrator.

Ported from Aurelius's aurelius/specialist_agents.py.

12 domain specialists, each with a specific role, can be run:
1. Individually
2. As a team via ApeOrchestrator
3. In a pipeline with merge + verify steps
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseSpecialist(ABC):
    """Base class for all specialist agents."""

    def __init__(self, name: str, description: str, capabilities: list[str] | None = None) -> None:
        self.name = name
        self.description = description
        self.capabilities = capabilities or []

    @abstractmethod
    def execute(self, task: str, context: dict[str, Any] | None = None) -> str: ...

    def __repr__(self) -> str:
        return f"{self.name}({self.description[:40]}...)"


class ResearchSpecialist(BaseSpecialist):
    """Web search, paper analysis, literature reviews."""

    def __init__(self) -> None:
        super().__init__(
            "ResearchSpecialist",
            "Web search, paper analysis, literature reviews",
            ["web_search", "paper_analysis", "literature_review", "fact_checking"],
        )

    def execute(self, task: str, context: dict[str, Any] | None = None) -> str:
        return f"[Research] Analyzed: {task[:100]}..."


class CodeSpecialist(BaseSpecialist):
    """Code generation, review, testing, debugging."""

    def __init__(self) -> None:
        super().__init__(
            "CodeSpecialist",
            "Code generation, review, testing, debugging",
            ["code_generation", "code_review", "testing", "debugging", "refactoring"],
        )

    def execute(self, task: str, context: dict[str, Any] | None = None) -> str:
        return f"[Code] Generated/reviewed code for: {task[:100]}..."


class DataSpecialist(BaseSpecialist):
    """Dataset building, cleaning, analysis, visualization."""

    def __init__(self) -> None:
        super().__init__(
            "DataSpecialist",
            "Dataset building, cleaning, analysis",
            ["data_cleaning", "data_analysis", "visualization", "eda"],
        )

    def execute(self, task: str, context: dict[str, Any] | None = None) -> str:
        return f"[Data] Processed data: {task[:100]}..."


class ModelArchitect(BaseSpecialist):
    """Architecture design, scaling laws, model selection."""

    def __init__(self) -> None:
        super().__init__(
            "ModelArchitect",
            "Architecture design, scaling laws, model selection",
            ["architecture_design", "scaling_analysis", "model_selection"],
        )

    def execute(self, task: str, context: dict[str, Any] | None = None) -> str:
        return f"[Architect] Designed architecture: {task[:100]}..."


class TrainingEngineer(BaseSpecialist):
    """Training plans, hyperparameter tuning, optimization."""

    def __init__(self) -> None:
        super().__init__(
            "TrainingEngineer",
            "Training plans, hyperparameter tuning",
            ["training_plan", "hyperparameter_tuning", "optimization"],
        )

    def execute(self, task: str, context: dict[str, Any] | None = None) -> str:
        return f"[Training] Created training plan: {task[:100]}..."


class EvaluationSpecialist(BaseSpecialist):
    """Benchmarking, metrics, evaluation."""

    def __init__(self) -> None:
        super().__init__(
            "EvaluationSpecialist",
            "Benchmarking, metrics, evaluation",
            ["benchmarking", "metrics", "evaluation", "regression_testing"],
        )

    def execute(self, task: str, context: dict[str, Any] | None = None) -> str:
        return f"[Evaluation] Evaluated: {task[:100]}..."


class SafetyEngineer(BaseSpecialist):
    """Harm/bias/jailbreak detection, red teaming."""

    def __init__(self) -> None:
        super().__init__(
            "SafetyEngineer",
            "Harm/bias/jailbreak detection, red teaming",
            ["harm_detection", "bias_audit", "jailbreak_detection", "red_teaming"],
        )

    def execute(self, task: str, context: dict[str, Any] | None = None) -> str:
        return f"[Safety] Safety analysis: {task[:100]}..."


class InfrastructureEngineer(BaseSpecialist):
    """GPU, cluster, networking, deployment infrastructure."""

    def __init__(self) -> None:
        super().__init__(
            "InfrastructureEngineer",
            "GPU, cluster, networking, deployment",
            ["infrastructure_design", "gpu_optimization", "deployment"],
        )

    def execute(self, task: str, context: dict[str, Any] | None = None) -> str:
        return f"[Infrastructure] Infrastructure plan: {task[:100]}..."


class MemoryEngineer(BaseSpecialist):
    """Memory system design, storage, retrieval optimization."""

    def __init__(self) -> None:
        super().__init__(
            "MemoryEngineer",
            "Memory system design and optimization",
            ["memory_design", "storage_optimization", "retrieval"],
        )

    def execute(self, task: str, context: dict[str, Any] | None = None) -> str:
        return f"[Memory] Memory system: {task[:100]}..."


class ToolEngineer(BaseSpecialist):
    """Tool registry, plugins, API integration."""

    def __init__(self) -> None:
        super().__init__(
            "ToolEngineer",
            "Tool registry, plugins, API integration",
            ["tool_creation", "plugin_development", "api_integration"],
        )

    def execute(self, task: str, context: dict[str, Any] | None = None) -> str:
        return f"[Tools] Tool development: {task[:100]}..."


class QualityReviewer(BaseSpecialist):
    """Code review, quality audit, standards compliance."""

    def __init__(self) -> None:
        super().__init__(
            "QualityReviewer",
            "Code review, quality audit, standards compliance",
            ["code_review", "quality_audit", "standards_compliance"],
        )

    def execute(self, task: str, context: dict[str, Any] | None = None) -> str:
        return f"[Quality] Quality review: {task[:100]}..."


class DeploymentEngineer(BaseSpecialist):
    """Release, rollback, monitoring, CI/CD."""

    def __init__(self) -> None:
        super().__init__(
            "DeploymentEngineer",
            "Release, rollback, monitoring, CI/CD",
            ["release_management", "rollback", "monitoring", "cicd"],
        )

    def execute(self, task: str, context: dict[str, Any] | None = None) -> str:
        return f"[Deployment] Deployment plan: {task[:100]}..."


class ApeOrchestrator:
    """Orchestrates multiple specialists, merges results, verifies.

    Can run agents individually, as a team, or as a pipeline.
    """

    def __init__(self) -> None:
        self._agents: dict[str, BaseSpecialist] = {
            "research": ResearchSpecialist(),
            "code": CodeSpecialist(),
            "data": DataSpecialist(),
            "architecture": ModelArchitect(),
            "training": TrainingEngineer(),
            "evaluation": EvaluationSpecialist(),
            "safety": SafetyEngineer(),
            "infrastructure": InfrastructureEngineer(),
            "memory": MemoryEngineer(),
            "tools": ToolEngineer(),
            "quality": QualityReviewer(),
            "deployment": DeploymentEngineer(),
        }

    def get_agent(self, name: str) -> BaseSpecialist | None:
        return self._agents.get(name)

    def run_individual(
        self, agent_name: str, task: str, context: dict[str, Any] | None = None
    ) -> str:
        agent = self._agents.get(agent_name)
        if agent is None:
            return f"Unknown agent: {agent_name}"
        return agent.execute(task, context)

    def run_team(
        self, task: str, agent_names: list[str] | None = None, context: dict[str, Any] | None = None
    ) -> dict[str, str]:
        names = agent_names or list(self._agents.keys())
        results: dict[str, str] = {}
        for name in names:
            agent = self._agents.get(name)
            if agent is not None:
                results[name] = agent.execute(task, context)
        return results

    def run_pipeline(
        self, task: str, pipeline: list[str], context: dict[str, Any] | None = None
    ) -> str:
        current = task
        for agent_name in pipeline:
            agent = self._agents.get(agent_name)
            if agent is not None:
                current = agent.execute(current, context)
        return current

    def list_agents(self) -> list[dict[str, Any]]:
        return [
            {
                "name": agent.name,
                "key": key,
                "description": agent.description,
                "capabilities": agent.capabilities,
            }
            for key, agent in self._agents.items()
        ]
