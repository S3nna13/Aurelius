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
]
