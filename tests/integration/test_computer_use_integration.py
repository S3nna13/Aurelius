"""Integration tests for the computer_use surface end-to-end pipeline.

Parse JSON accessibility tree → predict GUI action → verify action.
Also covers BrowserDriver lifecycle and Trajectory record/replay.

Inspired by OpenDevin/OpenDevin (browser tool), MoonshotAI/Kimi-Dev (coding agent loop),
Apache-2.0, clean-room reimplementation.
"""

from __future__ import annotations

import pytest

from src.computer_use.action_verifier import ActionVerifier, verify_trajectory
from src.computer_use.browser_driver import (
    BrowserDriverError,
    BrowserState,
    StubBrowserDriver,
)
from src.computer_use.gui_action import (
    ActionType,
    GUIAction,
    RuleBasedPredictor,
    get_action_predictor,
)
from src.computer_use.screen_parser import (
    AccessibilityNode,
    JSONTreeParser,
    ScreenSnapshot,
    get_screen_parser,
)
from src.computer_use.trajectory_replay import (
    TrajectoryRecorder,
    TrajectoryReplayer,
)

# ---------------------------------------------------------------------------
# Shared fixture
# ---------------------------------------------------------------------------

SAMPLE_TREE: dict = {
    "width": 1280,
    "height": 800,
    "root": {
        "role": "application",
        "name": "MyApp",
        "children": [
            {
                "role": "toolbar",
                "name": "Toolbar",
                "children": [
                    {
                        "role": "button",
                        "name": "New File",
                        "bbox": [10, 5, 80, 24],
                    },
                    {
                        "role": "button",
                        "name": "Open Folder",
                        "bbox": [100, 5, 100, 24],
                    },
                    {
                        "role": "button",
                        "name": "Save",
                        "bbox": [210, 5, 60, 24],
                    },
                ],
            },
            {
                "role": "editor",
                "name": "Code Editor",
                "bbox": [0, 40, 1280, 760],
                "attributes": {"editable": True},
            },
        ],
    },
    "ocr_text": "New File  Open Folder  Save",
}


# ---------------------------------------------------------------------------
# Integration tests
# ---------------------------------------------------------------------------


class TestComputerUseEndToEnd:
    """Full pipeline: parse → predict → verify."""

    def test_parse_json_tree_succeeds(self):
        parser = JSONTreeParser()
        snap = parser.parse(SAMPLE_TREE)
        assert isinstance(snap, ScreenSnapshot)
        assert snap.width == 1280
        assert snap.height == 800

    def test_predict_action_for_save_goal(self):
        parser = JSONTreeParser()
        snap = parser.parse(SAMPLE_TREE)
        predictor = RuleBasedPredictor()
        actions = predictor.predict(snap, "Save the document")
        assert len(actions) == 1
        assert actions[0].action_type == ActionType.CLICK
        assert actions[0].target_selector == "Save"

    def test_predict_action_for_new_file_goal(self):
        parser = JSONTreeParser()
        snap = parser.parse(SAMPLE_TREE)
        predictor = RuleBasedPredictor()
        actions = predictor.predict(snap, "New File")
        assert len(actions) == 1
        assert actions[0].target_selector == "New File"

    def test_verify_predicted_action_passes(self):
        parser = JSONTreeParser()
        snap = parser.parse(SAMPLE_TREE)
        predictor = RuleBasedPredictor()
        actions = predictor.predict(snap, "Save the document")
        assert len(actions) == 1
        verifier = ActionVerifier()
        ok, reason = verifier.verify(actions[0], snap)
        assert ok is True, f"Expected verification to pass, got: {reason}"

    def test_verify_trajectory_all_pass(self):
        parser = JSONTreeParser()
        snap = parser.parse(SAMPLE_TREE)
        actions = [
            GUIAction(action_type=ActionType.CLICK, target_selector="Save"),
            GUIAction(action_type=ActionType.CLICK, target_selector="New File"),
        ]
        results = verify_trajectory(actions, snap)
        assert len(results) == 2
        assert all(ok for ok, _ in results)

    def test_predict_no_match_returns_empty_or_wait(self):
        parser = JSONTreeParser()
        snap = parser.parse(SAMPLE_TREE)
        predictor = RuleBasedPredictor()
        actions = predictor.predict(snap, "Xyzzy Quux Nonexistent")
        assert isinstance(actions, list)

    def test_registries_accessible_via_package(self):
        import src.computer_use as cu

        assert "json_tree" in cu.SCREEN_PARSER_REGISTRY
        assert "rule_based" in cu.GUI_ACTION_REGISTRY

    def test_get_parser_from_registry_and_parse(self):
        parser_cls = get_screen_parser("json_tree")
        snap = parser_cls().parse(SAMPLE_TREE)
        assert snap.root_node.name == "MyApp"

    def test_get_predictor_from_registry_and_predict(self):
        parser = JSONTreeParser()
        snap = parser.parse(SAMPLE_TREE)
        predictor_cls = get_action_predictor("rule_based")
        actions = predictor_cls().predict(snap, "Open Folder")
        assert len(actions) == 1
        assert actions[0].target_selector == "Open Folder"

    def test_denied_action_fails_verification(self):
        parser = JSONTreeParser()
        snap = parser.parse(SAMPLE_TREE)
        bad_action = GUIAction(action_type=ActionType.CLICK, target_selector="delete everything")
        verifier = ActionVerifier()
        ok, reason = verifier.verify(bad_action, snap)
        assert ok is False

    def test_snapshot_ocr_text_preserved(self):
        parser = JSONTreeParser()
        snap = parser.parse(SAMPLE_TREE)
        assert snap.ocr_text == "New File  Open Folder  Save"

    def test_editor_node_attributes_preserved(self):
        parser = JSONTreeParser()
        snap = parser.parse(SAMPLE_TREE)
        editor = snap.root_node.children[1]
        assert editor.attributes.get("editable") is True


# ---------------------------------------------------------------------------
# BrowserDriver lifecycle integration
# ---------------------------------------------------------------------------


class TestBrowserDriverIntegration:
    """End-to-end lifecycle test for StubBrowserDriver."""

    def test_browser_driver_stub(self):
        """navigate → click → type_text → get_state full lifecycle."""
        driver = StubBrowserDriver()

        # Navigate
        state = driver.navigate("https://example.com")
        assert isinstance(state, BrowserState)
        assert state.url == "https://example.com"
        assert state.ready is True

        # Click
        state = driver.click("#submit-btn")
        assert isinstance(state, BrowserState)
        assert state.ready is True

        # Type text
        state = driver.type_text("#search-input", "Aurelius")
        assert isinstance(state, BrowserState)
        assert state.ready is True
        assert state.html_snapshot is not None
        assert "Aurelius" in state.html_snapshot

        # get_state reflects the latest state
        current = driver.get_state()
        assert current.url == "https://example.com"

        # close
        driver.close()
        with pytest.raises(BrowserDriverError):
            driver.get_state()


# ---------------------------------------------------------------------------
# Trajectory record and replay integration
# ---------------------------------------------------------------------------


class TestTrajectoryIntegration:
    """Record a 2-step trajectory and replay it end-to-end."""

    def test_trajectory_record_and_replay(self):
        """Record 2 steps, finalize, replay with StubBrowserDriver, verify all ok."""
        # Set up a stub driver and navigate so click/type_text will work.
        driver = StubBrowserDriver()
        driver.navigate("https://example.com")

        # Record two steps.
        recorder = TrajectoryRecorder(goal="search for something")

        snap0 = ScreenSnapshot(
            width=1280,
            height=800,
            root_node=AccessibilityNode(role="application", name="App"),
        )
        step0 = recorder.record_step(
            GUIAction(action_type=ActionType.CLICK, target_selector="#search-btn"),
            snapshot_before=snap0,
        )

        snap1 = ScreenSnapshot(
            width=1280,
            height=800,
            root_node=AccessibilityNode(role="application", name="App"),
        )
        step1 = recorder.record_step(
            GUIAction(
                action_type=ActionType.TYPE,
                target_selector="#search-input",
                value="Aurelius",
            ),
            snapshot_before=snap1,
        )

        assert step0.step_id == 0
        assert step1.step_id == 1

        traj = recorder.finalize(success=True)
        assert traj.success is True
        assert len(traj.steps) == 2

        # Replay.
        replayer = TrajectoryReplayer()
        statuses = replayer.replay(traj, driver)

        assert len(statuses) == 2
        assert all("ok" in s for s in statuses)

        # Verify structural integrity.
        ok, issues = replayer.verify(traj)
        assert ok is True
        assert issues == []


# ---------------------------------------------------------------------------
# WebArena harness integration
# ---------------------------------------------------------------------------


class TestWebArenaHarnessIntegration:
    """End-to-end integration: register a task, run with StubBrowserDriver."""

    def test_webarena_harness_run(self):
        """Register a task, run with a stub agent, verify TaskResult structure."""
        from src.computer_use.webarena_eval import (
            TaskResult,
            WebArenaHarness,
            WebTask,
        )

        harness = WebArenaHarness()
        task = WebTask(
            task_id="integration_nav",
            description="Navigate to a target URL.",
            start_url="https://example.com",
            success_criteria=["UNREACHABLE_CRITERION_XYZ"],
            max_steps=3,
            tags=["integration"],
        )
        harness.register_task(task)

        driver = StubBrowserDriver()
        call_count = {"n": 0}

        def stub_agent(state, task):
            # Return None after 1 call so the loop exits early.
            if call_count["n"] >= 1:
                return None
            call_count["n"] += 1
            return GUIAction(action_type=ActionType.CLICK, target_selector="#btn")

        result = harness.run_task("integration_nav", driver, stub_agent)

        assert isinstance(result, TaskResult)
        assert result.task_id == "integration_nav"
        # Agent ran one step then returned None — steps_taken should be 1.
        assert result.steps_taken == 1
        # Criterion is unreachable, so success must be False.
        assert result.success is False
        # No exception should have been raised.
        assert result.error is None
