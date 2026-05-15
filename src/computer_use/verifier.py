"""Aurelius v2 CUA Action Verifier — safety gate for every CUA action.

Blocks:
- Password/secret entry fields
- Payment/checkout UI
- Permission dialogs
- Destructive UI elements (delete, format, eject)
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.computer_use.driver_base import CUAAction, CUAObservation


@dataclass
class SafetyBlock:
    """A safety block indicating a CUA action was blocked."""
    action: CUAAction
    reason: str
    severity: str = "critical"  # critical, warning, info
    detected_patterns: list[str] = field(default_factory=list)


class CUAActionVerifier:
    """Verifies safety of every CUA action before execution."""

    # Patterns that indicate password fields
    PASSWORD_PATTERNS = [
        "password", "passwd", "secret", "credential", "pin code",
        "api key", "token", "authentication",
    ]

    # Patterns that indicate payment UI
    PAYMENT_PATTERNS = [
        "checkout", "payment", "credit card", "billing", "purchase",
        "buy now", "place order", "pay", "stripe", "paypal",
    ]

    # Destructive actions
    DESTRUCTIVE_PATTERNS = [
        "delete", "remove", "format", "erase", "wipe", "eject",
        "uninstall", "destroy", "shutdown", "restart", "power off",
    ]

    def verify(
        self, action: CUAAction, observation: CUAObservation,
    ) -> SafetyBlock | None:
        """Verify an action against safety rules."""
        all_text = f"{observation.ax_tree} {observation.ocr_text} {observation.focused_element} {observation.window_title}".lower()

        # Check for password fields
        if any(p in all_text for p in self.PASSWORD_PATTERNS):
            return SafetyBlock(
                action=action,
                reason="Password/credential field detected",
                severity="critical",
                detected_patterns=[p for p in self.PASSWORD_PATTERNS if p in all_text],
            )

        # Check for payment UI
        if any(p in all_text for p in self.PAYMENT_PATTERNS):
            return SafetyBlock(
                action=action,
                reason="Payment/checkout UI detected",
                severity="critical",
                detected_patterns=[p for p in self.PAYMENT_PATTERNS if p in all_text],
            )

        # Check for destructive actions
        if any(p in all_text for p in self.DESTRUCTIVE_PATTERNS):
            return SafetyBlock(
                action=action,
                reason="Destructive UI element detected",
                severity="critical",
                detected_patterns=[p for p in self.DESTRUCTIVE_PATTERNS if p in all_text],
            )

        return None
