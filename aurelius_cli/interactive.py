"""Aurelius CLI v2 — Interactive chat/agent modes with status bar.

Handles the interactive REPL loop, slash commands, and mode switching.
"""

from __future__ import annotations

import logging
import sys
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Mode definitions
# ---------------------------------------------------------------------------

VALID_MODES = {
    "chat",
    "code",
    "agent",
    "computer",
    "research",
    "operator",
    "training",
    "safe",
}

DEFAULT_MODE = "chat"


@dataclass
class ModeState:
    """Tracks the current interactive mode and its parameters."""

    mode: str = DEFAULT_MODE
    model: str = "forge"
    backend: str = ""
    profile: str = "default"


# ---------------------------------------------------------------------------
# Slash command definitions
# ---------------------------------------------------------------------------

SLASH_COMMANDS: dict[str, str] = {
    "/help": "Show available commands",
    "/status": "Show current model/backend/profile/RAM-VRAM status",
    "/model": "Switch model (swift/forge/atlas)",
    "/backend": "Switch backend",
    "/profile": "Switch profile",
    "/hardware": "Show detected hardware",
    "/capabilities": "Show current capability report",
    "/memory": "Search runtime memory",
    "/skills": "List native skills",
    "/skill": "Show skill details",
    "/tools": "List available tools",
    "/cua": "CUA capture/mode",
    "/checkpoint": "Create workspace checkpoint",
    "/rollback": "Rollback to last checkpoint",
    "/daies": "Run DAIES validation",
    "/export": "Start model export",
    "/serve": "Runtime status",
    "/ui": "Open Mission Control",
    "/logs": "Show recent logs",
    "/traces": "Show recent traces",
    "/config": "View current configuration",
    "/mode": "Switch mode",
    "/quit": "Exit interactive session",
}


# ---------------------------------------------------------------------------
# Permission card
# ---------------------------------------------------------------------------

@dataclass
class PermissionRequest:
    """Represents a permission request for a risky action."""

    action: str
    target: str
    risk: str  # low | medium | high
    checkpoint: bool = False


def format_permission_card(req: PermissionRequest) -> str:
    """Format a permission card for display."""
    lines = [
        "Permission required",
        f"Action: {req.action}",
        f"Target: {req.target}",
        f"Risk: {req.risk}",
    ]
    if req.checkpoint:
        lines.append("Checkpoint: will create before action")
    lines.append("Approve? (yes / no / always-this-session)")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Interactive session
# ---------------------------------------------------------------------------

class InteractiveSession:
    """Manages an interactive CLI session with status bar and slash commands."""

    def __init__(
        self,
        runtime_client: Any = None,
        config: Any = None,
        mode: str = DEFAULT_MODE,
        status_bar_renderer: Any = None,
    ) -> None:
        self.runtime_client = runtime_client
        self.config = config
        self.mode_state = ModeState(mode=mode if mode in VALID_MODES else DEFAULT_MODE)
        self.status_bar_renderer = status_bar_renderer
        self._running = False
        self._session_approved: set[str] = set()  # Actions approved for this session

    # -- Public interface -----------------------------------------------------

    def run(self) -> int:
        """Run the interactive session loop.

        Returns:
            Exit code (0 = normal exit, non-zero = error).
        """
        self._running = True
        self._print_banner()
        self._print_welcome()

        while self._running:
            try:
                user_input = self._get_input()
                if not user_input:
                    continue
                if user_input.startswith("/"):
                    self._handle_slash_command(user_input.strip())
                else:
                    self._handle_user_message(user_input)
            except KeyboardInterrupt:
                self._print("\nInterrupted. Type /quit to exit.")
            except EOFError:
                self._running = False

        self._print_exit()
        return 0

    def request_permission(self, req: PermissionRequest) -> bool:
        """Ask user for permission on a risky action.

        Returns:
            True if action is approved.
        """
        session_key = f"{req.action}:{req.target}"
        if session_key in self._session_approved:
            return True

        self._print(format_permission_card(req))

        try:
            response = self._get_input().strip().lower()
        except (KeyboardInterrupt, EOFError):
            return False

        if response in ("yes", "y"):
            return True
        if response in ("always-this-session", "always"):
            self._session_approved.add(session_key)
            return True
        return False

    # -- Internal helpers -----------------------------------------------------

    def _print(self, text: str) -> None:
        """Print text to the terminal."""
        print(text, flush=True)

    def _print_banner(self) -> None:
        """Print the Aurelius banner."""
        banner = [
            "",
            "  ╔═══════════════════════════════════════╗",
            "  ║     Aurelius CLI v2 — Interactive     ║",
            "  ╚═══════════════════════════════════════╝",
            "",
        ]
        self._print("\n".join(banner))

    def _print_welcome(self) -> None:
        """Print welcome message with current status."""
        self._print(f"  Mode: {self.mode_state.mode}")
        self._print(f"  Model: {self.mode_state.model}")
        self._print(f"  Type /help for commands, /quit to exit")
        self._print("")

    def _print_exit(self) -> None:
        """Print exit message."""
        self._print("\n  Goodbye!")

    def _get_input(self) -> str:
        """Get user input. Tries prompt_toolkit first, falls back to input()."""
        try:
            return input(f"\n[{self.mode_state.mode}]> ")
        except (EOFError, KeyboardInterrupt) as exc:
            raise EOFError from exc

    def _handle_slash_command(self, command: str) -> None:
        """Handle a slash command."""
        parts = command.split(None, 1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        handler = {
            "/help": lambda: self._cmd_help(),
            "/status": lambda: self._cmd_status(),
            "/mode": lambda: self._cmd_mode(args),
            "/model": lambda: self._cmd_model(args),
            "/backend": lambda: self._cmd_backend(args),
            "/profile": lambda: self._cmd_profile(args),
            "/hardware": lambda: self._cmd_hardware(),
            "/capabilities": lambda: self._cmd_capabilities(),
            "/memory": lambda: self._cmd_memory(args),
            "/skills": lambda: self._cmd_skills(args),
            "/skill": lambda: self._cmd_skill(args),
            "/tools": lambda: self._cmd_tools(args),
            "/cua": lambda: self._cmd_cua(args),
            "/checkpoint": lambda: self._cmd_checkpoint(),
            "/rollback": lambda: self._cmd_rollback(),
            "/daies": lambda: self._cmd_daies(args),
            "/export": lambda: self._cmd_export(args),
            "/serve": lambda: self._cmd_serve(),
            "/ui": lambda: self._cmd_ui(),
            "/logs": lambda: self._cmd_logs(args),
            "/traces": lambda: self._cmd_traces(args),
            "/config": lambda: self._cmd_config(),
            "/quit": lambda: self._cmd_quit(),
        }.get(cmd)

        if handler:
            handler()
        else:
            self._print(f"Unknown command: {cmd}")
            self._print("Type /help for available commands.")

    def _handle_user_message(self, message: str) -> None:
        """Handle a regular user message (chat/agent/etc)."""
        if self.runtime_client:
            self._print(f"\n  [Processing in {self.mode_state.mode} mode...]")
            self._print(f"  Model: {self.mode_state.model}")
            self._print(f"  (Full inference requires runtime backend)")
        else:
            self._print(f"\n  [{self.mode_state.mode}] Received: {message[:80]}...")
            self._print("  (Connect to runtime backend for full inference)")

    # -- Slash command implementations ----------------------------------------

    def _cmd_help(self) -> None:
        """Show available slash commands."""
        self._print("\n  Available slash commands:\n")
        max_len = max(len(cmd) for cmd in SLASH_COMMANDS)
        for cmd, desc in sorted(SLASH_COMMANDS.items()):
            self._print(f"  {cmd:<{max_len}}  {desc}")
        self._print("")

    def _cmd_status(self) -> None:
        """Show current runtime status."""
        if self.runtime_client:
            status = self.runtime_client.get_runtime_status(self.config)
            self._print(f"\n  Model: {status.model_name}")
            self._print(f"  Backend: {status.backend}")
            self._print(f"  Context: {status.context_length}")
            self._print(f"  RAM: {status.ram_used_gb:.1f}/{status.ram_total_gb:.1f} GB")
            if status.vram_total_gb > 0:
                self._print(f"  VRAM: {status.vram_used_gb:.1f}/{status.vram_total_gb:.1f} GB")
            self._print(f"  Skills: {status.skill_count}")
            self._print(f"  Tools: {status.tool_count}")
            self._print(f"  Profile: {status.profile_id}")
            self._print(f"  CUA: {status.cua_mode}")
        else:
            self._print("\n  No runtime client connected.")

    def _cmd_mode(self, args: str) -> None:
        """Switch mode."""
        if not args:
            self._print(f"\n  Current mode: {self.mode_state.mode}")
            self._print(f"  Available: {', '.join(sorted(VALID_MODES))}")
            return
        mode = args.strip().lower()
        if mode in VALID_MODES:
            self.mode_state.mode = mode
            self._print(f"\n  Mode switched to: {mode}")
        else:
            self._print(f"\n  Unknown mode: {mode}")
            self._print(f"  Available: {', '.join(sorted(VALID_MODES))}")

    def _cmd_model(self, args: str) -> None:
        """Switch model."""
        if not args:
            self._print(f"\n  Current model: {self.mode_state.model}")
            return
        self.mode_state.model = args.strip().lower()
        self._print(f"\n  Model switched to: {self.mode_state.model}")

    def _cmd_backend(self, args: str) -> None:
        """Switch backend."""
        if not args:
            self._print(f"\n  Current backend: {self.mode_state.backend or 'auto'}")
            return
        self.mode_state.backend = args.strip()
        self._print(f"\n  Backend switched to: {self.mode_state.backend}")

    def _cmd_profile(self, args: str) -> None:
        """Switch profile."""
        if not args:
            self._print(f"\n  Current profile: {self.mode_state.profile}")
            return
        self.mode_state.profile = args.strip()
        self._print(f"\n  Profile switched to: {self.mode_state.profile}")

    def _cmd_hardware(self) -> None:
        """Show detected hardware."""
        if self.runtime_client:
            hw = self.runtime_client.detect_hardware()
            self._print(f"\n  System: {hw.system} ({hw.machine})")
            self._print(f"  CPU cores: {hw.cpu_count}")
            self._print(f"  RAM: {hw.ram_total_gb:.1f} GB")
            self._print(f"  GPU: {hw.gpu_name or 'None detected'}")
            if hw.gpu_vram_gb > 0:
                self._print(f"  GPU VRAM: {hw.gpu_vram_gb:.1f} GB")
            self._print(f"  CUDA: {'Yes' if hw.cuda_available else 'No'}")
            self._print(f"  MPS: {'Yes' if hw.mps_available else 'No'}")
            self._print(f"  MLX: {'Yes' if hw.mlx_available else 'No'}")
            self._print(f"  Unified memory: {'Yes' if hw.unified_memory else 'No'}")
        else:
            self._print("\n  No runtime client connected.")

    def _cmd_capabilities(self) -> None:
        """Show capability report."""
        if self.runtime_client:
            report = self.runtime_client.capability_report()
            self._print(f"\n  Capability Report:")
            for key, val in report.items():
                self._print(f"    {key}: {val}")
        else:
            self._print("\n  No runtime client connected.")

    def _cmd_memory(self, args: str) -> None:
        """Search runtime memory."""
        if not args:
            self._print("\n  Usage: /memory search <query>")
            return
        self._print(f"\n  Searching memory for: {args}")
        self._print("  (Memory search requires runtime backend)")

    def _cmd_skills(self, args: str) -> None:
        """List skills."""
        if self.runtime_client:
            skills = self.runtime_client.list_skills()
            self._print(f"\n  Skills ({len(skills)} total):")
            categories: dict[str, list[Any]] = {}
            for skill in skills:
                categories.setdefault(skill.category, []).append(skill)
            for cat, cat_skills in sorted(categories.items()):
                self._print(f"    [{cat}]")
                for s in cat_skills[:5]:
                    status = "✓" if s.enabled else "✗"
                    self._print(f"      {status} {s.name}")
                if len(cat_skills) > 5:
                    self._print(f"      ... and {len(cat_skills) - 5} more")
        else:
            self._print("\n  No runtime client connected.")

    def _cmd_skill(self, args: str) -> None:
        """Show skill details."""
        if not args:
            self._print("\n  Usage: /skill <id>")
            return
        self._print(f"\n  Looking up skill: {args}")
        self._print("  (Skill lookup requires runtime backend)")

    def _cmd_tools(self, args: str) -> None:
        """List tools."""
        if self.runtime_client:
            tools = self.runtime_client.list_tools()
            self._print(f"\n  Tools ({len(tools)} total):")
            for t in tools[:20]:
                perm = " (requires permission)" if t.requires_permission else ""
                status = "✓" if t.enabled else "✗"
                self._print(f"    {status} {t.name}{perm}")
            if len(tools) > 20:
                self._print(f"    ... and {len(tools) - 20} more")
        else:
            self._print("\n  No runtime client connected.")

    def _cmd_cua(self, args: str) -> None:
        """CUA commands."""
        if args == "capture":
            self._print("\n  CUA capture initiated...")
            self._print("  (CUA requires runtime backend)")
        elif args == "mode":
            self._print(f"\n  CUA mode: {self.config.cua.mode if self.config else 'unknown'}")
        else:
            self._print("\n  Usage: /cua capture | /cua mode")

    def _cmd_checkpoint(self) -> None:
        """Create workspace checkpoint."""
        self._print("\n  Creating checkpoint...")
        self._print("  (Checkpoints require runtime backend)")

    def _cmd_rollback(self) -> None:
        """Rollback to last checkpoint."""
        self._print("\n  Rolling back to last checkpoint...")
        self._print("  (Rollback requires runtime backend)")

    def _cmd_daies(self, args: str) -> None:
        """Run DAIES validation."""
        gate = args.strip().lower() if args else ""
        if gate == "quick":
            self._print("\n  Running DAIES quick gate check...")
        elif gate == "full":
            self._print("\n  Running DAIES full validation...")
        else:
            self._print("\n  Usage: /daies quick | /daies full")

    def _cmd_export(self, args: str) -> None:
        """Start model export."""
        fmt = args.strip().lower() if args else ""
        self._print(f"\n  Export format: {fmt or 'auto'}")
        self._print("  (Export requires runtime backend)")

    def _cmd_serve(self) -> None:
        """Runtime status."""
        if self.runtime_client:
            status = self.runtime_client.get_runtime_status(self.config)
            self._print(f"\n  Runtime Status:")
            self._print(f"    Model: {status.model_name}")
            self._print(f"    Backend: {status.backend}")
            self._print(f"    Session count: {status.session_count}")
        else:
            self._print("\n  Runtime not connected.")

    def _cmd_ui(self) -> None:
        """Open Mission Control."""
        self._print("\n  Opening Mission Control...")
        self._print("  (UI requires the frontend to be built)")

    def _cmd_logs(self, args: str) -> None:
        """Show recent logs."""
        lines = int(args) if args.isdigit() else 20
        self._print(f"\n  Showing last {lines} log entries...")
        self._print("  (Logs require logging to be configured)")

    def _cmd_traces(self, args: str) -> None:
        """Show recent traces."""
        lines = int(args) if args.isdigit() else 10
        self._print(f"\n  Showing last {lines} trace entries...")
        self._print("  (Traces require observability to be configured)")

    def _cmd_config(self) -> None:
        """View current configuration."""
        if self.config:
            self._print("\n  Current Configuration:")
            self._print(f"    Model: {self.config.model.name}")
            self._print(f"    Family: {self.config.model.family}")
            self._print(f"    Backend: {self.config.model.backend or 'auto'}")
            self._print(f"    Context length: {self.config.model.context_length}")
            self._print(f"    Profile: {self.config.runtime.profile}")
            self._print(f"    Memory policy: {self.config.runtime.memory_policy}")
            self._print(f"    Temperature: {self.config.model.temperature}")
            self._print(f"    Top P: {self.config.model.top_p}")
            self._print(f"    Max tokens: {self.config.model.max_tokens}")
            self._print(f"    Skills preload: {self.config.skills.preload_count}")
            self._print(f"    CUA enabled: {self.config.cua.enabled}")
            self._print(f"    CUA mode: {self.config.cua.mode}")
            self._print(f"    MCP servers: {len(self.config.mcp.servers)}")
        else:
            self._print("\n  No configuration loaded.")

    def _cmd_quit(self) -> None:
        """Exit the session."""
        self._print("\n  Exiting...")
        self._running = False
