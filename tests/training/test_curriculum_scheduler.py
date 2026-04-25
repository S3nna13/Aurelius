"""Tests for src/training/curriculum_scheduler.py."""

import pytest
from src.training.curriculum_scheduler import (
    CurriculumScheduler,
    CurriculumStage,
    DifficultyLevel,
)


# ---------------------------------------------------------------------------
# DifficultyLevel enum
# ---------------------------------------------------------------------------

class TestDifficultyLevel:
    def test_easy_value(self):
        assert DifficultyLevel.EASY == "easy"

    def test_medium_value(self):
        assert DifficultyLevel.MEDIUM == "medium"

    def test_hard_value(self):
        assert DifficultyLevel.HARD == "hard"

    def test_expert_value(self):
        assert DifficultyLevel.EXPERT == "expert"

    def test_four_members(self):
        assert len(DifficultyLevel) == 4

    def test_is_str(self):
        assert isinstance(DifficultyLevel.EASY, str)


# ---------------------------------------------------------------------------
# CurriculumStage dataclass
# ---------------------------------------------------------------------------

class TestCurriculumStage:
    def test_stage_id(self):
        s = CurriculumStage(stage_id=0, difficulty=DifficultyLevel.EASY)
        assert s.stage_id == 0

    def test_difficulty_field(self):
        s = CurriculumStage(stage_id=1, difficulty=DifficultyLevel.HARD)
        assert s.difficulty == DifficultyLevel.HARD

    def test_default_min_accuracy(self):
        s = CurriculumStage(stage_id=0, difficulty=DifficultyLevel.EASY)
        assert s.min_accuracy == 0.7

    def test_default_max_steps(self):
        s = CurriculumStage(stage_id=0, difficulty=DifficultyLevel.EASY)
        assert s.max_steps == 1000

    def test_default_description(self):
        s = CurriculumStage(stage_id=0, difficulty=DifficultyLevel.EASY)
        assert s.description == ""

    def test_custom_min_accuracy(self):
        s = CurriculumStage(stage_id=0, difficulty=DifficultyLevel.EASY, min_accuracy=0.9)
        assert s.min_accuracy == 0.9

    def test_custom_max_steps(self):
        s = CurriculumStage(stage_id=0, difficulty=DifficultyLevel.EASY, max_steps=500)
        assert s.max_steps == 500

    def test_custom_description(self):
        s = CurriculumStage(stage_id=0, difficulty=DifficultyLevel.EASY, description="intro")
        assert s.description == "intro"


# ---------------------------------------------------------------------------
# CurriculumScheduler
# ---------------------------------------------------------------------------

class TestCurriculumSchedulerInit:
    def test_default_has_four_stages(self):
        cs = CurriculumScheduler()
        # We can count by advancing through all stages
        count = 0
        while True:
            count += 1
            advanced = cs.advance(1.0)
            if not advanced:
                break
        assert count == 4  # 3 advances + 1 failed = 4 stages total

    def test_default_starts_easy(self):
        cs = CurriculumScheduler()
        assert cs.current_stage().difficulty == DifficultyLevel.EASY

    def test_custom_stages(self):
        stages = [
            CurriculumStage(0, DifficultyLevel.EASY),
            CurriculumStage(1, DifficultyLevel.HARD),
        ]
        cs = CurriculumScheduler(stages=stages)
        assert cs.current_stage().difficulty == DifficultyLevel.EASY


class TestCurrentStage:
    def test_starts_at_stage_0(self):
        cs = CurriculumScheduler()
        assert cs.current_stage().stage_id == 0

    def test_stage_0_is_easy(self):
        cs = CurriculumScheduler()
        assert cs.current_stage().difficulty == DifficultyLevel.EASY


class TestAdvance:
    def test_advance_returns_true_on_sufficient_accuracy(self):
        cs = CurriculumScheduler()
        assert cs.advance(0.7) is True

    def test_advance_returns_true_above_min_accuracy(self):
        cs = CurriculumScheduler()
        assert cs.advance(0.9) is True

    def test_advance_returns_false_below_min_accuracy(self):
        cs = CurriculumScheduler()
        assert cs.advance(0.5) is False

    def test_advance_moves_to_next_stage(self):
        cs = CurriculumScheduler()
        cs.advance(1.0)
        assert cs.current_stage().difficulty == DifficultyLevel.MEDIUM

    def test_advance_progresses_through_all_stages(self):
        cs = CurriculumScheduler()
        cs.advance(1.0)
        cs.advance(1.0)
        cs.advance(1.0)
        assert cs.current_stage().difficulty == DifficultyLevel.EXPERT

    def test_advance_at_last_stage_returns_false(self):
        cs = CurriculumScheduler()
        cs.advance(1.0)
        cs.advance(1.0)
        cs.advance(1.0)
        assert cs.advance(1.0) is False

    def test_advance_resets_step_count(self):
        cs = CurriculumScheduler()
        cs.record_step()
        cs.record_step()
        cs.advance(1.0)
        assert cs.step_count() == 0

    def test_advance_does_not_move_on_low_accuracy(self):
        cs = CurriculumScheduler()
        cs.advance(0.1)
        assert cs.current_stage().difficulty == DifficultyLevel.EASY


class TestStepCount:
    def test_step_count_starts_zero(self):
        cs = CurriculumScheduler()
        assert cs.step_count() == 0

    def test_record_step_increments(self):
        cs = CurriculumScheduler()
        cs.record_step()
        assert cs.step_count() == 1

    def test_multiple_record_steps(self):
        cs = CurriculumScheduler()
        for _ in range(5):
            cs.record_step()
        assert cs.step_count() == 5

    def test_record_step_with_accuracy_arg(self):
        cs = CurriculumScheduler()
        cs.record_step(accuracy=0.8)
        assert cs.step_count() == 1


class TestShouldAdvance:
    def test_should_advance_false_before_max_steps(self):
        cs = CurriculumScheduler()
        for _ in range(999):
            cs.record_step()
        assert cs.should_advance() is False

    def test_should_advance_true_at_max_steps(self):
        cs = CurriculumScheduler()
        for _ in range(1000):
            cs.record_step()
        assert cs.should_advance() is True

    def test_should_advance_true_beyond_max_steps(self):
        cs = CurriculumScheduler()
        for _ in range(1500):
            cs.record_step()
        assert cs.should_advance() is True

    def test_should_advance_custom_max_steps(self):
        stages = [
            CurriculumStage(0, DifficultyLevel.EASY, max_steps=5),
            CurriculumStage(1, DifficultyLevel.HARD, max_steps=5),
        ]
        cs = CurriculumScheduler(stages=stages)
        for _ in range(5):
            cs.record_step()
        assert cs.should_advance() is True


class TestProgress:
    def test_progress_returns_dict(self):
        cs = CurriculumScheduler()
        p = cs.progress()
        assert isinstance(p, dict)

    def test_progress_has_stage_key(self):
        cs = CurriculumScheduler()
        assert "stage" in cs.progress()

    def test_progress_has_total_stages_key(self):
        cs = CurriculumScheduler()
        assert "total_stages" in cs.progress()

    def test_progress_has_difficulty_key(self):
        cs = CurriculumScheduler()
        assert "difficulty" in cs.progress()

    def test_progress_has_steps_key(self):
        cs = CurriculumScheduler()
        assert "steps" in cs.progress()

    def test_progress_stage_starts_zero(self):
        cs = CurriculumScheduler()
        assert cs.progress()["stage"] == 0

    def test_progress_total_stages_four(self):
        cs = CurriculumScheduler()
        assert cs.progress()["total_stages"] == 4

    def test_progress_difficulty_easy_initially(self):
        cs = CurriculumScheduler()
        assert cs.progress()["difficulty"] == "easy"

    def test_progress_steps_matches_step_count(self):
        cs = CurriculumScheduler()
        cs.record_step()
        cs.record_step()
        assert cs.progress()["steps"] == 2

    def test_progress_updates_after_advance(self):
        cs = CurriculumScheduler()
        cs.advance(1.0)
        p = cs.progress()
        assert p["stage"] == 1
        assert p["difficulty"] == "medium"
