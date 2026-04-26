"""Onboarding / empty-state flow for the Aurelius terminal UI surface.

Inspired by MoonshotAI/kimi-cli (MIT, terminal session lifecycle),
Anthropic Claude Code (MIT, command palette UX), clean-room
reimplementation with original Aurelius branding.

Provides a step-based onboarding model that renders via Rich Panels and
tracks completion state.  Only rich, stdlib, and project-local imports
are used.
"""

from __future__ import annotations

from dataclasses import dataclass

from rich.console import Console
from rich.panel import Panel
from rich.text import Text


@dataclass
class OnboardingStep:
    """A single step in an onboarding flow.

    Attributes:
        id: Unique step identifier within its :class:`OnboardingFlow`.
        title: Short step title shown in the header.
        description: Longer description of what the user should do.
        action_label: Optional call-to-action label (e.g. ``"Press Enter"``).
        completed: Whether this step has been completed.
    """

    id: str
    title: str
    description: str
    action_label: str | None = None
    completed: bool = False


class OnboardingFlow:
    """A sequence of :class:`OnboardingStep` objects.

    State mutations (advance, mark complete) go through methods.

    Args:
        steps: Ordered list of :class:`OnboardingStep` objects.

    Raises:
        ValueError: If *steps* is empty.
    """

    def __init__(self, steps: list[OnboardingStep]) -> None:
        if not steps:
            raise ValueError("OnboardingFlow requires at least one step")
        self.steps: list[OnboardingStep] = list(steps)
        self.current_step_index: int = 0

    # ------------------------------------------------------------------
    # State transitions
    # ------------------------------------------------------------------

    def advance(self) -> OnboardingStep | None:
        """Mark the current step completed and advance to the next.

        Returns:
            The new current :class:`OnboardingStep`, or ``None`` if the
            flow is already complete after marking the final step done.
        """
        if self.current_step_index < len(self.steps):
            self.steps[self.current_step_index].completed = True
            self.current_step_index += 1
        if self.current_step_index < len(self.steps):
            return self.steps[self.current_step_index]
        return None

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def is_complete(self) -> bool:
        """``True`` when every step has been completed."""
        return all(s.completed for s in self.steps)

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    def render(self, console: Console) -> None:
        """Render the onboarding flow as a Rich Panel to *console*."""
        total = len(self.steps)
        done = sum(1 for s in self.steps if s.completed)

        if self.is_complete:
            body = Text("All steps complete!", style="bold green")
            title = f"Aurelius Setup — {done}/{total} complete"
            console.print(Panel(body, title=title, border_style="green"))
            return

        current = (
            self.steps[self.current_step_index]
            if self.current_step_index < total
            else self.steps[-1]
        )

        progress_bar = _render_progress_bar(done, total, width=30)
        lines = Text()
        lines.append(f"{progress_bar}  {done}/{total} steps complete\n\n", style="dim")
        lines.append(f"Step {self.current_step_index + 1}: {current.title}\n", style="bold")
        lines.append(f"{current.description}\n", style="")
        if current.action_label:
            lines.append(f"\n{current.action_label}", style="italic cyan")

        title = "Aurelius Onboarding"
        console.print(Panel(lines, title=title, border_style="cyan"))


def _render_progress_bar(done: int, total: int, width: int = 20) -> str:
    """Return a simple ASCII progress bar string."""
    if total == 0:
        return "[" + " " * width + "]"
    filled = int(done / total * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


ONBOARDING_REGISTRY: dict[str, OnboardingFlow] = {}

# ---------------------------------------------------------------------------
# Pre-registered "first_run" flow
# ---------------------------------------------------------------------------

_FIRST_RUN_STEPS: list[OnboardingStep] = [
    OnboardingStep(
        id="welcome",
        title="Welcome",
        description=(
            "Welcome to Aurelius — a keyboard-first terminal LLM interface.\n"
            "Press Enter or type '/' to open the command palette."
        ),
        action_label="Press Enter to continue",
    ),
    OnboardingStep(
        id="configure-model",
        title="Configure model",
        description=(
            "Choose which language model to use. You can update this setting "
            "at any time via the command palette (toggle-motion, show-branding)."
        ),
        action_label="Press Enter to continue",
    ),
    OnboardingStep(
        id="choose-backend",
        title="Choose backend",
        description=(
            "Select the inference backend: local, remote API, or hybrid.\n"
            "Aurelius supports multiple backends simultaneously."
        ),
        action_label="Press Enter to continue",
    ),
    OnboardingStep(
        id="start-session",
        title="Start session",
        description=(
            "Everything is set up! Type a message or use '/' to open the\n"
            "command palette and explore available commands."
        ),
        action_label="Press Enter to begin",
    ),
]

ONBOARDING_REGISTRY["first_run"] = OnboardingFlow(_FIRST_RUN_STEPS)

__all__ = [
    "OnboardingStep",
    "OnboardingFlow",
    "ONBOARDING_REGISTRY",
]
