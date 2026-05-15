"""CLI v2 configuration management — load/save YAML config with profile inheritance."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)

# Config search paths (in priority order)
_CONFIG_DIRS = [
    Path(os.environ.get("AURELIUS_CONFIG_DIR", "")) if os.environ.get("AURELIUS_CONFIG_DIR") else None,
    Path.home() / ".config" / "aurelius",
    Path.home() / ".aurelius",
]

_DEFAULT_CONFIG_FILENAME = "config.yaml"
_DEFAULT_PROFILE = "default"


@dataclass
class ModelConfig:
    """Model selection and inference parameters."""

    name: str = "forge"
    family: str = "forge"  # swift | forge | atlas
    backend: str = ""
    quantization: str = ""
    context_length: int = 32768
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 2048
    seed: int | None = None


@dataclass
class RuntimeConfig:
    """Runtime execution parameters."""

    profile: str = _DEFAULT_PROFILE
    capability_mode: str = ""  # full_local | reduced_local | remote | verifier_only | split
    memory_policy: str = "balanced"  # conservative | balanced | performance | frontier
    max_ram_gb: float = 0.0  # 0 = auto-detect
    max_vram_gb: float = 0.0  # 0 = auto-detect
    gpu_device: int = 0
    threads: int = 0  # 0 = auto


@dataclass
class SkillConfig:
    """Skill system configuration."""

    preload_count: int = 10
    enabled: list[str] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)


@dataclass
class ToolConfig:
    """Tool system configuration."""

    enabled: list[str] = field(default_factory=list)
    disabled: list[str] = field(default_factory=list)
    permission_mode: str = "prompt"  # prompt | always_this_session | always


@dataclass
class CUAConfig:
    """Computer-use agent configuration."""

    enabled: bool = False
    mode: str = "local_full"  # local_full | local_safe | remote | verifier_only
    capture_interval: float = 1.0
    safety_level: str = "high"  # high | medium | low


@dataclass
class MCPServers:
    """MCP server connection configuration."""

    servers: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class UISettings:
    """UI/Mission Control settings."""

    theme: str = "dark"
    status_bar: bool = True
    confirmations: bool = True
    show_capabilities: bool = True


@dataclass
class LogConfig:
    """Logging configuration."""

    level: str = "INFO"
    file: str = ""
    max_size_mb: int = 50
    rotation_count: int = 5


@dataclass
class AureliusConfig:
    """Top-level Aurelius configuration."""

    version: str = "2.0.0"
    model: ModelConfig = field(default_factory=ModelConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    skills: SkillConfig = field(default_factory=SkillConfig)
    tools: ToolConfig = field(default_factory=ToolConfig)
    cua: CUAConfig = field(default_factory=CUAConfig)
    mcp: MCPServers = field(default_factory=MCPServers)
    ui: UISettings = field(default_factory=UISettings)
    logs: LogConfig = field(default_factory=LogConfig)
    custom: dict[str, Any] = field(default_factory=dict)


def _resolve_config_dir() -> Path:
    """Find the configuration directory."""
    # Check AURELIUS_CONFIG_DIR override first
    env_dir = os.environ.get("AURELIUS_CONFIG_DIR")
    if env_dir:
        return Path(env_dir)

    for d in _CONFIG_DIRS:
        if d and d.exists():
            return d

    # Create default config directory
    default = Path.home() / ".config" / "aurelius"
    default.mkdir(parents=True, exist_ok=True)
    return default


def _resolve_config_path(config_path: str | Path | None = None) -> Path:
    """Resolve the full config file path."""
    if config_path:
        return Path(config_path)

    config_dir = _resolve_config_dir()
    return config_dir / _DEFAULT_CONFIG_FILENAME


def load_config(
    config_path: str | Path | None = None,
    profile: str | None = None,
) -> AureliusConfig:
    """Load configuration from YAML file with profile support.

    Args:
        config_path: Explicit path to config file. If None, uses default search order.
        profile: Profile name to apply on top of base config.

    Returns:
        Loaded and validated AureliusConfig instance.
    """
    path = _resolve_config_path(config_path)
    logger.debug("Loading config from %s", path)

    if not path.exists():
        logger.info("No config file found at %s, using defaults", path)
        return AureliusConfig()

    try:
        with open(path) as f:
            raw: dict[str, Any] = yaml.safe_load(f) or {}
    except (yaml.YAMLError, OSError) as e:
        logger.warning("Failed to load config from %s: %s — using defaults", path, e)
        return AureliusConfig()

    # Extract base config (minus profiles)
    base_raw = {k: v for k, v in raw.items() if k != "profiles"}
    config = _dict_to_config(base_raw)

    # Apply profile overrides
    effective_profile = profile or config.runtime.profile
    profiles: dict[str, dict[str, Any]] | None = raw.get("profiles")
    if profiles and effective_profile in profiles:
        logger.debug("Applying profile: %s", effective_profile)
        profile_overrides = profiles[effective_profile]
        config = _merge_config(config, _dict_to_config(profile_overrides))

    return config


def save_config(config: AureliusConfig, config_path: str | Path | None = None) -> Path:
    """Save configuration to YAML file.

    Args:
        config: Config object to save.
        config_path: Explicit output path. If None, uses default search order.

    Returns:
        Path to the saved config file.
    """
    path = _resolve_config_path(config_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    raw = _config_to_dict(config)
    with open(path, "w") as f:
        yaml.dump(raw, f, default_flow_style=False, sort_keys=False, Dumper=yaml.SafeDumper)

    logger.info("Config saved to %s", path)
    return path


def _dict_to_config(raw: dict[str, Any]) -> AureliusConfig:
    """Convert a raw dict to an AureliusConfig, ignoring unknown keys."""
    model = ModelConfig(**raw.get("model", {}))
    runtime = RuntimeConfig(**raw.get("runtime", {}))
    skills = SkillConfig(**raw.get("skills", {}))
    tools = ToolConfig(**raw.get("tools", {}))
    cua = CUAConfig(**raw.get("cua", {}))
    mcp = MCPServers(**raw.get("mcp", {}))
    ui = UISettings(**raw.get("ui", {}))
    logs = LogConfig(**raw.get("logs", {}))
    custom = raw.get("custom", {})

    return AureliusConfig(
        version=raw.get("version", "2.0.0"),
        model=model,
        runtime=runtime,
        skills=skills,
        tools=tools,
        cua=cua,
        mcp=mcp,
        ui=ui,
        logs=logs,
        custom=custom,
    )


def _config_to_dict(config: AureliusConfig) -> dict[str, Any]:
    """Convert AureliusConfig to a serializable dict."""
    return {
        "version": config.version,
        "model": {
            "name": config.model.name,
            "family": config.model.family,
            "backend": config.model.backend,
            "quantization": config.model.quantization,
            "context_length": config.model.context_length,
            "temperature": config.model.temperature,
            "top_p": config.model.top_p,
            "max_tokens": config.model.max_tokens,
            "seed": config.model.seed,
        },
        "runtime": {
            "profile": config.runtime.profile,
            "capability_mode": config.runtime.capability_mode,
            "memory_policy": config.runtime.memory_policy,
            "max_ram_gb": config.runtime.max_ram_gb,
            "max_vram_gb": config.runtime.max_vram_gb,
            "gpu_device": config.runtime.gpu_device,
            "threads": config.runtime.threads,
        },
        "skills": {
            "preload_count": config.skills.preload_count,
            "enabled": config.skills.enabled,
            "disabled": config.skills.disabled,
        },
        "tools": {
            "enabled": config.tools.enabled,
            "disabled": config.tools.disabled,
            "permission_mode": config.tools.permission_mode,
        },
        "cua": {
            "enabled": config.cua.enabled,
            "mode": config.cua.mode,
            "capture_interval": config.cua.capture_interval,
            "safety_level": config.cua.safety_level,
        },
        "mcp": {"servers": config.mcp.servers},
        "ui": {
            "theme": config.ui.theme,
            "status_bar": config.ui.status_bar,
            "confirmations": config.ui.confirmations,
            "show_capabilities": config.ui.show_capabilities,
        },
        "logs": {
            "level": config.logs.level,
            "file": config.logs.file,
            "max_size_mb": config.logs.max_size_mb,
            "rotation_count": config.logs.rotation_count,
        },
        "custom": config.custom,
    }


def _merge_config(base: AureliusConfig, override: AureliusConfig) -> AureliusConfig:
    """Merge override config on top of base. Non-empty override values win."""
    # Merge model
    model_fields: dict[str, Any] = {}
    for fld in ("name", "family", "backend", "quantization", "context_length",
                "temperature", "top_p", "max_tokens", "seed"):
        ov = getattr(override.model, fld)
        # Only use override if it's non-default or explicitly set
        if ov != getattr(ModelConfig(), fld):
            model_fields[fld] = ov
        else:
            model_fields[fld] = getattr(base.model, fld)

    # Merge runtime
    runtime_fields: dict[str, Any] = {}
    for fld in ("profile", "capability_mode", "memory_policy", "max_ram_gb",
                "max_vram_gb", "gpu_device", "threads"):
        ov = getattr(override.runtime, fld)
        default = getattr(RuntimeConfig(), fld)
        if ov != default and ov != 0 and ov != "":
            runtime_fields[fld] = ov
        else:
            runtime_fields[fld] = getattr(base.runtime, fld)

    return AureliusConfig(
        version=override.version if override.version != "2.0.0" else base.version,
        model=ModelConfig(**model_fields),
        runtime=RuntimeConfig(**runtime_fields),
        skills=override.skills if override.skills.enabled or override.skills.disabled else base.skills,
        tools=override.tools if override.tools.enabled or override.tools.disabled else base.tools,
        cua=override.cua if override.cua.enabled or override.cua.mode != "local_full" else base.cua,
        mcp=override.mcp if override.mcp.servers else base.mcp,
        ui=override.ui if override.ui.theme != "dark" else base.ui,
        logs=override.logs if override.logs.level != "INFO" else base.logs,
        custom={**base.custom, **override.custom},
    )
