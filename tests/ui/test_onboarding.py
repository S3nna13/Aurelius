"""Tests for src.ui.onboarding."""

from __future__ import annotations

import pytest
from rich.console import Console

from src.ui.onboarding import (
    ONBOARDING_REGISTRY,
    OnboardingFlow,
    OnboardingStep,
)


# ---------------------------------------------------------------------------
# Registry population
# ---------------------------------------------------------------------------


def test_first_run_in_registry() -> None:
    assert "first_run" in ONBOARDING_REGISTRY


def test_first_run_has_four_steps() -> None:
    flow = ONBOARDING_REGISTRY["first_run"]
    assert len(flow.steps) == 4


def test_first_run_step_ids() -> None:
    flow = ONBOARDING_REGISTRY["first_run"]
    ids = [s.id for s in flow.steps]
    assert "welcome" in ids
    assert "configure-model" in ids
    assert "choose-backend" in ids
    assert "start-session" in ids


# ---------------------------------------------------------------------------
# advance
# ---------------------------------------------------------------------------


def _fresh_flow() -> OnboardingFlow:
    """Return a fresh 4-step flow for each test (avoids shared-state issues)."""
    return OnboardingFlow([
        OnboardingStep(id="s1", title="Step 1", description="desc 1"),
        OnboardingStep(id="s2", title="Step 2", description="desc 2"),
        OnboardingStep(id="s3", title="Step 3", description="desc 3"),
        OnboardingStep(id="s4", title="Step 4", description="desc 4"),
    ])


def test_advance_marks_current_step_completed() -> None:
    flow = _fresh_flow()
    assert not flow.steps[0].completed
    flow.advance()
    assert flow.steps[0].completed


def test_advance_returns_next_step() -> None:
    flow = _fresh_flow()
    next_step = flow.advance()
    assert next_step is not None
    assert next_step.id == "s2"


def test_advance_returns_none_after_last_step() -> None:
    flow = _fresh_flow()
    flow.advance()  # s1 -> s2
    flow.advance()  # s2 -> s3
    flow.advance()  # s3 -> s4
    result = flow.advance()  # s4 -> done
    assert result is None


def test_advance_increments_current_step_index() -> None:
    flow = _fresh_flow()
    assert flow.current_step_index == 0
    flow.advance()
    assert flow.current_step_index == 1


# ---------------------------------------------------------------------------
# is_complete
# ---------------------------------------------------------------------------


def test_is_complete_false_at_start() -> None:
    flow = _fresh_flow()
    assert not flow.is_complete


def test_is_complete_false_mid_flow() -> None:
    flow = _fresh_flow()
    flow.advance()
    flow.advance()
    assert not flow.is_complete


def test_is_complete_true_after_all_advance() -> None:
    flow = _fresh_flow()
    flow.advance()
    flow.advance()
    flow.advance()
    flow.advance()
    assert flow.is_complete


# ---------------------------------------------------------------------------
# render
# ---------------------------------------------------------------------------


def test_render_initial_state_does_not_crash() -> None:
    flow = _fresh_flow()
    console = Console(record=True)
    flow.render(console)


def test_render_complete_flow_does_not_crash() -> None:
    flow = _fresh_flow()
    for _ in range(4):
        flow.advance()
    console = Console(record=True)
    flow.render(console)


def test_render_output_contains_step_title() -> None:
    flow = _fresh_flow()
    console = Console(record=True)
    flow.render(console)
    output = console.export_text()
    assert "Step 1" in output


def test_render_complete_shows_all_done() -> None:
    flow = _fresh_flow()
    for _ in range(4):
        flow.advance()
    console = Console(record=True)
    flow.render(console)
    output = console.export_text()
    assert "complete" in output.lower()


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_empty_steps_raises() -> None:
    with pytest.raises(ValueError):
        OnboardingFlow([])


def test_flow_with_single_step() -> None:
    flow = OnboardingFlow([OnboardingStep(id="only", title="Only", description="d")])
    assert not flow.is_complete
    result = flow.advance()
    assert result is None
    assert flow.is_complete


# ---------------------------------------------------------------------------
# ONBOARDING_REGISTRY
# ---------------------------------------------------------------------------


def test_onboarding_registry_is_dict() -> None:
    assert isinstance(ONBOARDING_REGISTRY, dict)
