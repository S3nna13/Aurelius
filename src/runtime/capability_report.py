"""Aurelius v2 Capability Report — generates reports showing requested vs actual model/backend/artifact."""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import Any

from src.runtime.profile_schema import CapabilityMode


class CapabilityStatus(enum.StrEnum):
    """Status of each capability on the current profile."""
    LIVE = "live"
    REMOTE = "remote"
    DEGRADED = "degraded"
    DISABLED = "disabled"
    SIMULATED = "simulated"
    UNAVAILABLE = "unavailable"


@dataclass
class CapabilityInfo:
    """Information about a single capability."""
    name: str
    status: CapabilityStatus
    details: str = ""
    local: bool = False


@dataclass
class CapabilityReport:
    """Full capability report showing what is live, remote, degraded, disabled, or simulated.

    Implements the no-silent-fallback rule: every report must clearly distinguish
    between requested and actual model, backend, artifact, and execution mode.
    """

    requested_model: str
    actual_model: str
    execution_mode: CapabilityMode
    backend: str
    artifact: str
    quantization: str
    context_budget: int
    local_or_remote: str
    hardware_profile: str
    capabilities: dict[str, CapabilityInfo] = field(default_factory=dict)
    disabled_capabilities: list[str] = field(default_factory=list)
    fallback_reason: str | None = None
    live_status: str = "live"
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert report to a JSON-serialisable dict."""
        caps = {}
        for name, info in self.capabilities.items():
            caps[name] = {
                "status": info.status.value,
                "details": info.details,
                "local": info.local,
            }
        return {
            "requested_model": self.requested_model,
            "actual_model": self.actual_model,
            "execution_mode": self.execution_mode.value,
            "backend": self.backend,
            "artifact": self.artifact,
            "quantization": self.quantization,
            "context_budget": self.context_budget,
            "local_or_remote": self.local_or_remote,
            "hardware_profile": self.hardware_profile,
            "capabilities": caps,
            "disabled_capabilities": self.disabled_capabilities,
            "fallback_reason": self.fallback_reason,
            "live_status": self.live_status,
            "metadata": self.metadata,
        }

    def is_live(self) -> bool:
        """Return True only if the requested model + actual model match and status is live."""
        return (
            self.live_status == "live"
            and self.requested_model == self.actual_model
            and not self.fallback_reason
        )

    def has_silent_fallback(self) -> bool:
        """Check for silent fallback violation."""
        return self.requested_model != self.actual_model and not self.fallback_reason

    def assert_no_silent_fallback(self) -> None:
        """Raise if a silent fallback is detected."""
        if self.has_silent_fallback():
            raise SilentFallbackError(
                f"Requested model={self.requested_model} but running actual_model={self.actual_model}"
                f" without fallback_reason. Silent fallback is prohibited."
            )

    @classmethod
    def create_full_local(
        cls,
        model: str = "forge",
        backend: str = "mlx",
        artifact: str = "forge-q4-mlx",
        quantization: str = "q4",
        context: int = 32768,
        hardware: str = "mac_silicon_32gb",
    ) -> CapabilityReport:
        """Create a report for a fully-local, live configuration."""
        caps = {
            "chat": CapabilityInfo("chat", CapabilityStatus.LIVE, "Interactive chat", True),
            "tools": CapabilityInfo("tools", CapabilityStatus.LIVE, "Tool execution", True),
            "native_skills": CapabilityInfo("native_skills", CapabilityStatus.LIVE, "Native skill library", True),
            "memory": CapabilityInfo("memory", CapabilityStatus.LIVE, "Runtime memory", True),
            "cua": CapabilityInfo("cua", CapabilityStatus.LIVE, "Local computer use", True),
            "long_context": CapabilityInfo("long_context", CapabilityStatus.LIVE, f"Context up to {context} tokens", True),
            "multimodal": CapabilityInfo("multimodal", CapabilityStatus.LIVE, "Vision/document perception", True),
            "cron": CapabilityInfo("cron", CapabilityStatus.LIVE, "Scheduled jobs", True),
            "delegation": CapabilityInfo("delegation", CapabilityStatus.LIVE, "Subagent spawning", True),
            "checkpoints": CapabilityInfo("checkpoints", CapabilityStatus.LIVE, "Workspace snapshots", True),
            "export": CapabilityInfo("export", CapabilityStatus.LIVE, "Model artifact export", True),
        }
        return cls(
            requested_model=model,
            actual_model=model,
            execution_mode=CapabilityMode.FULL_LOCAL,
            backend=backend,
            artifact=artifact,
            quantization=quantization,
            context_budget=context,
            local_or_remote="local",
            hardware_profile=hardware,
            capabilities=caps,
        )

    @classmethod
    def create_remote_only(
        cls,
        requested: str = "atlas",
        actual: str = "atlas",
        backend: str = "remote_aurelius",
        context: int = 128000,
        hardware: str = "jetson_nano_4gb",
    ) -> CapabilityReport:
        """Create a report for a controller-only with remote inference configuration."""
        caps = {
            "chat": CapabilityInfo("chat", CapabilityStatus.REMOTE, "Remote model inference", False),
            "tools": CapabilityInfo("tools", CapabilityStatus.LIVE, "Local tool execution", True),
            "native_skills": CapabilityInfo("native_skills", CapabilityStatus.LIVE, "Local skill evaluation", True),
            "memory": CapabilityInfo("memory", CapabilityStatus.LIVE, "Local runtime memory", True),
            "cua": CapabilityInfo("cua", CapabilityStatus.DEGRADED, "Verifier-only CUA; remote action execution", False),
            "long_context": CapabilityInfo("long_context", CapabilityStatus.REMOTE, f"Remote context up to {context} tokens", False),
            "multimodal": CapabilityInfo("multimodal", CapabilityStatus.REMOTE, "Remote vision", False),
            "cron": CapabilityInfo("cron", CapabilityStatus.LIVE, "Local scheduler", True),
            "delegation": CapabilityInfo("delegation", CapabilityStatus.LIVE, "Local subagent spawning", True),
            "checkpoints": CapabilityInfo("checkpoints", CapabilityStatus.LIVE, "Workspace snapshots", True),
            "export": CapabilityInfo("export", CapabilityStatus.UNAVAILABLE, "No local model to export", False),
        }
        disabled = ["local_full_cua", "local_vlm"]
        return cls(
            requested_model=requested,
            actual_model=actual,
            execution_mode=CapabilityMode.CONTROLLER_ONLY,
            backend=backend,
            artifact="atlas-fp8-trtllm",
            quantization="fp8",
            context_budget=context,
            local_or_remote="remote_model_local_tools",
            hardware_profile=hardware,
            capabilities=caps,
            disabled_capabilities=disabled,
        )


class SilentFallbackError(Exception):
    """Raised when model downgrade is not explicitly reported."""
    pass
