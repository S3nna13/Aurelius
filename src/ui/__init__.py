"""Aurelius terminal/IDE UI surface — original branding, keyboard-first, reduced-motion aware."""

from __future__ import annotations

from src.ui.branding import (
    DEFAULT_BRANDING,
    MASCOT_ASCII,
    AureliusBranding,
)
from src.ui.errors import UIError
from src.ui.motion import (
    MOTION_REGISTRY,
    MotionSpec,
    get_motion,
    list_motions,
    play,
    register_motion,
)
from src.ui.panel_layout import (
    PANEL_LAYOUT_REGISTRY,
    PanelLayout,
    compose_layout,
    get_panel_layout,
    list_panel_layouts,
    register_panel_layout,
)
from src.ui.ui_surface import (
    UI_SURFACE_REGISTRY,
    UISurface,
    get_ui_surface,
    list_ui_surfaces,
    register_ui_surface,
)
from src.ui.aurelius_shell import (
    AureliusShell,
    AureliusShellError,
    MessageEnvelope,
    SkillRecord,
    Workstream,
    WorkflowRun,
    WorkflowStep,
)
from src.ui.welcome import WelcomePanel, render_welcome
from src.ui.command_palette import (
    CommandEntry,
    CommandPalette,
    CommandPaletteError,
    CommandPaletteState,
    COMMAND_PALETTE_REGISTRY,
)
from src.ui.status_hierarchy import (
    StatusLevel,
    StatusState,
    StatusNode,
    StatusTree,
    STATUS_TREE_REGISTRY,
)
from src.ui.keyboard_nav import (
    KeyBinding,
    KeyBindingError,
    KeyboardNav,
    KEYBOARD_NAV_REGISTRY,
)
from src.ui.onboarding import (
    OnboardingStep,
    OnboardingFlow,
    ONBOARDING_REGISTRY,
)

# Register new UI surfaces into the surface registry.
_COMMAND_PALETTE_SURFACE = UISurface(
    surface_id="command-palette",
    title="Command Palette",
    panels=("palette",),
    default_layout="stoic-focus",
    keyboard_map={"esc": "cancel", "/": "search"},
)
_STATUS_HIERARCHY_SURFACE = UISurface(
    surface_id="status-hierarchy",
    title="Status Hierarchy",
    panels=("status",),
    default_layout="stoic-focus",
    keyboard_map={"esc": "cancel", "up": "nav_up", "down": "nav_down"},
)
_KEYBOARD_NAV_SURFACE = UISurface(
    surface_id="keyboard-nav",
    title="Keyboard Navigation",
    panels=("help",),
    default_layout="stoic-focus",
    keyboard_map={"esc": "cancel"},
)
_ONBOARDING_SURFACE = UISurface(
    surface_id="onboarding",
    title="Onboarding",
    panels=("onboarding",),
    default_layout="stoic-focus",
    keyboard_map={"esc": "cancel", "enter": "confirm"},
)

for _surface in (
    _COMMAND_PALETTE_SURFACE,
    _STATUS_HIERARCHY_SURFACE,
    _KEYBOARD_NAV_SURFACE,
    _ONBOARDING_SURFACE,
):
    if _surface.surface_id not in UI_SURFACE_REGISTRY:
        register_ui_surface(_surface)

__all__ = [
    "UIError",
    "AureliusBranding",
    "DEFAULT_BRANDING",
    "MASCOT_ASCII",
    "MotionSpec",
    "MOTION_REGISTRY",
    "play",
    "register_motion",
    "get_motion",
    "list_motions",
    "PanelLayout",
    "PANEL_LAYOUT_REGISTRY",
    "compose_layout",
    "register_panel_layout",
    "get_panel_layout",
    "list_panel_layouts",
    "UISurface",
    "UI_SURFACE_REGISTRY",
    "register_ui_surface",
    "get_ui_surface",
    "list_ui_surfaces",
    "AureliusShell",
    "AureliusShellError",
    "MessageEnvelope",
    "SkillRecord",
    "Workstream",
    "WorkflowRun",
    "WorkflowStep",
    "WelcomePanel",
    "render_welcome",
    # Command palette
    "CommandEntry",
    "CommandPalette",
    "CommandPaletteError",
    "CommandPaletteState",
    "COMMAND_PALETTE_REGISTRY",
    # Status hierarchy
    "StatusLevel",
    "StatusState",
    "StatusNode",
    "StatusTree",
    "STATUS_TREE_REGISTRY",
    # Keyboard navigation
    "KeyBinding",
    "KeyBindingError",
    "KeyboardNav",
    "KEYBOARD_NAV_REGISTRY",
    # Onboarding
    "OnboardingStep",
    "OnboardingFlow",
    "ONBOARDING_REGISTRY",
]
