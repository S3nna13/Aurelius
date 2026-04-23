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
from src.ui.transcript_viewer import (
    TranscriptRole,
    TranscriptEntry,
    TranscriptViewer,
    TRANSCRIPT_VIEWER_REGISTRY,
    TranscriptViewerError,
)
from src.ui.diff_viewer import (
    DiffLine,
    DiffChunk,
    ParsedDiff,
    parse_unified_diff,
    DiffViewer,
    DiffViewerError,
)
from src.ui.task_panel import (
    TaskEntry,
    TaskPanel,
    TASK_PANEL_REGISTRY,
    TaskPanelError,
)
from src.ui.streaming_renderer import (
    TokenChunk,
    StreamingState,
    StreamingRenderer,
    STREAMING_RENDERER_REGISTRY,
    StreamingRendererError,
)
from src.ui.session_manager import (
    SessionState,
    AureliusSession,
    SessionManager,
    DEFAULT_SESSION_MANAGER,
    SESSION_MANAGER_REGISTRY,
    SessionManagerError,
)
from src.ui.debug_panel import (
    DebugMetric,
    DebugSection,
    DebugPanel,
    DebugPanelError,
    DEBUG_PANEL_REGISTRY,
)
from src.ui.progress_renderer import (
    ProgressTask,
    ETAEstimator,
    ProgressRenderer,
    ProgressError,
    PROGRESS_RENDERER_REGISTRY,
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
_TRANSCRIPT_VIEWER_SURFACE = UISurface(
    surface_id="transcript-viewer",
    title="Transcript Viewer",
    panels=("transcript",),
    default_layout="stoic-focus",
    keyboard_map={"esc": "cancel", "up": "scroll_up", "down": "scroll_down"},
)
_DIFF_VIEWER_SURFACE = UISurface(
    surface_id="diff-viewer",
    title="Diff Viewer",
    panels=("diff",),
    default_layout="stoic-focus",
    keyboard_map={"esc": "cancel", "up": "scroll_up", "down": "scroll_down"},
)
_TASK_PANEL_SURFACE = UISurface(
    surface_id="task-panel",
    title="Task Panel",
    panels=("tasks",),
    default_layout="stoic-focus",
    keyboard_map={"esc": "cancel", "up": "nav_up", "down": "nav_down"},
)
_STREAMING_RENDERER_SURFACE = UISurface(
    surface_id="streaming-renderer",
    title="Streaming Renderer",
    panels=("stream",),
    default_layout="stoic-focus",
    keyboard_map={"esc": "cancel"},
)
_SESSION_MANAGER_SURFACE = UISurface(
    surface_id="session-manager",
    title="Session Manager",
    panels=("sessions",),
    default_layout="stoic-focus",
    keyboard_map={"esc": "cancel", "tab": "next_session"},
)
_DEBUG_PANEL_SURFACE = UISurface(
    surface_id="debug-panel",
    title="Debug Panel",
    panels=("debug",),
    default_layout="stoic-focus",
    keyboard_map={"esc": "cancel", "c": "collapse"},
)
_PROGRESS_RENDERER_SURFACE = UISurface(
    surface_id="progress-renderer",
    title="Progress Renderer",
    panels=("progress",),
    default_layout="stoic-focus",
    keyboard_map={"esc": "cancel"},
)

for _surface in (
    _COMMAND_PALETTE_SURFACE,
    _STATUS_HIERARCHY_SURFACE,
    _KEYBOARD_NAV_SURFACE,
    _ONBOARDING_SURFACE,
    _TRANSCRIPT_VIEWER_SURFACE,
    _DIFF_VIEWER_SURFACE,
    _TASK_PANEL_SURFACE,
    _STREAMING_RENDERER_SURFACE,
    _SESSION_MANAGER_SURFACE,
    _DEBUG_PANEL_SURFACE,
    _PROGRESS_RENDERER_SURFACE,
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
    # Transcript viewer
    "TranscriptRole",
    "TranscriptEntry",
    "TranscriptViewer",
    "TRANSCRIPT_VIEWER_REGISTRY",
    "TranscriptViewerError",
    # Diff viewer
    "DiffLine",
    "DiffChunk",
    "ParsedDiff",
    "parse_unified_diff",
    "DiffViewer",
    "DiffViewerError",
    # Task panel
    "TaskEntry",
    "TaskPanel",
    "TASK_PANEL_REGISTRY",
    "TaskPanelError",
    # Streaming renderer
    "TokenChunk",
    "StreamingState",
    "StreamingRenderer",
    "STREAMING_RENDERER_REGISTRY",
    "StreamingRendererError",
    # Session manager
    "SessionState",
    "AureliusSession",
    "SessionManager",
    "DEFAULT_SESSION_MANAGER",
    "SESSION_MANAGER_REGISTRY",
    "SessionManagerError",
    # Debug panel
    "DebugMetric",
    "DebugSection",
    "DebugPanel",
    "DebugPanelError",
    "DEBUG_PANEL_REGISTRY",
    # Progress renderer
    "ProgressTask",
    "ETAEstimator",
    "ProgressRenderer",
    "ProgressError",
    "PROGRESS_RENDERER_REGISTRY",
]
