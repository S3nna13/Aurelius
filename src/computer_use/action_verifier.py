"""Action safety verifier for the Aurelius computer_use surface.

Inspired by OpenDevin/OpenDevin (browser tool), MoonshotAI/Kimi-Dev (coding agent loop),
Apache-2.0, clean-room reimplementation.

No playwright, pyautogui, or OS accessibility API imports.
"""

from __future__ import annotations

from src.computer_use.gui_action import ActionType, GUIAction
from src.computer_use.screen_parser import AccessibilityNode, ScreenSnapshot


# ---------------------------------------------------------------------------
# Deny list
# ---------------------------------------------------------------------------

VERIFIER_DENY_LIST: frozenset[str] = frozenset({
    "delete",
    "format",
    "shutdown",
    "rm",
    "rmdir",
    "drop",
    "wipe",
    "erase",
    "terminate",
    "kill",
    "poweroff",
    "reboot",
    "destroy",
})


# ---------------------------------------------------------------------------
# Verifier
# ---------------------------------------------------------------------------

class ActionVerifier:
    """Verifies the safety and feasibility of GUIActions against a ScreenSnapshot."""

    def verify(self, action: GUIAction, snapshot: ScreenSnapshot) -> tuple[bool, str]:
        """Verify a single action.

        Checks (in order):
        1. target_selector or value must not contain a deny-listed pattern.
        2. CLICK — target_selector must match a node in the accessibility tree.
        3. TYPE  — value must be non-None and non-empty.
        4. DRAG  — coords must be non-None.

        Parameters
        ----------
        action:
            The action to verify.
        snapshot:
            The current screen snapshot used for node lookup.

        Returns
        -------
        tuple[bool, str]
            ``(True, reason)`` if the action passes; ``(False, reason)`` otherwise.
        """
        # --- Deny-list check (covers selector, value, and metadata goal) ---
        candidates = [
            action.target_selector or "",
            action.value or "",
            action.metadata.get("goal", "") if action.metadata else "",
        ]
        for candidate in candidates:
            for pattern in VERIFIER_DENY_LIST:
                if pattern in candidate.lower():
                    return (
                        False,
                        f"Action denied: contains deny-listed pattern {pattern!r} in {candidate!r}",
                    )

        # --- Action-type-specific checks ---
        if action.action_type == ActionType.CLICK:
            if action.target_selector is None:
                return False, "CLICK action requires a non-None target_selector"
            if not self._node_exists(snapshot.root_node, action.target_selector):
                return (
                    False,
                    f"CLICK target {action.target_selector!r} not found in snapshot tree",
                )

        elif action.action_type == ActionType.TYPE:
            if action.value is None or action.value == "":
                return False, "TYPE action requires a non-None, non-empty value"

        elif action.action_type == ActionType.DRAG:
            if action.coords is None:
                return False, "DRAG action requires non-None coords"

        return True, "Action verified successfully"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _node_exists(self, node: AccessibilityNode, selector: str) -> bool:
        """Return True if any node in the tree has a name matching *selector*."""
        if node.name == selector:
            return True
        return any(self._node_exists(child, selector) for child in node.children)


# ---------------------------------------------------------------------------
# Trajectory verifier
# ---------------------------------------------------------------------------

def verify_trajectory(
    actions: list[GUIAction], snapshot: ScreenSnapshot
) -> list[tuple[bool, str]]:
    """Verify a sequence of actions against a single snapshot.

    Parameters
    ----------
    actions:
        List of actions to verify.
    snapshot:
        Screen snapshot to verify against.

    Returns
    -------
    list[tuple[bool, str]]
        One result per action; empty list when *actions* is empty.
    """
    verifier = ActionVerifier()
    return [verifier.verify(action, snapshot) for action in actions]
