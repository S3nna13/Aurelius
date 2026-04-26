"""Socratic questioning tutor inspired by DeepTutor.

Teaches by guiding the user through questions and hints rather than
giving direct answers.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


class SocraticTutorError(Exception):
    """Raised for invalid tutor operations."""


@dataclass
class ConceptMastery:
    """Track a user's mastery of a single concept."""

    concept_id: str
    name: str
    mastery_level: float = 0.0
    attempts: int = 0
    correct_streak: int = 0


@dataclass
class TutorSession:
    """A single tutoring session."""

    session_id: str
    user_id: str
    active_concept_id: str | None = None
    history: list[dict[str, Any]] = field(default_factory=list)


class SocraticTutor:
    """Socratic tutor that teaches via guided questioning."""

    def __init__(self, concepts: list[dict] | None = None) -> None:
        self._concepts: dict[str, dict] = {}
        self._mastery: dict[str, dict[str, ConceptMastery]] = {}
        self._sessions: dict[str, TutorSession] = {}
        self._session_question: dict[str, tuple[str, int]] = {}
        self._question_streaks: dict[str, dict[str, list[int]]] = {}

        for concept in concepts or []:
            self.add_concept(concept)

    # ------------------------------------------------------------------
    # Concept management
    # ------------------------------------------------------------------

    def add_concept(self, concept: dict) -> None:
        """Add a new concept at runtime."""
        cid = concept["concept_id"]
        self._concepts[cid] = concept
        for user_mastery in self._mastery.values():
            if cid not in user_mastery:
                user_mastery[cid] = ConceptMastery(concept_id=cid, name=concept["name"])
        for user_qs in self._question_streaks.values():
            if cid not in user_qs:
                num_questions = len(concept.get("questions", []))
                user_qs[cid] = [0] * num_questions

    # ------------------------------------------------------------------
    # Session management
    # ------------------------------------------------------------------

    def start_session(self, user_id: str) -> TutorSession:
        """Create a new tutoring session."""
        session_id = uuid.uuid4().hex[:8]
        session = TutorSession(session_id=session_id, user_id=user_id)
        self._sessions[session_id] = session
        return session

    def _ensure_user_mastery(self, user_id: str) -> None:
        """Initialise mastery and question-streak state for a user."""
        if user_id not in self._mastery:
            self._mastery[user_id] = {}
            self._question_streaks[user_id] = {}
            for cid, concept in self._concepts.items():
                self._mastery[user_id][cid] = ConceptMastery(concept_id=cid, name=concept["name"])
                num_questions = len(concept.get("questions", []))
                self._question_streaks[user_id][cid] = [0] * num_questions

    # ------------------------------------------------------------------
    # Question selection
    # ------------------------------------------------------------------

    def _prerequisites_met(self, user_id: str, concept: dict) -> bool:
        for prereq in concept.get("prerequisites", []):
            mastery = self._mastery.get(user_id, {}).get(prereq)
            if mastery is None or mastery.mastery_level < 0.8:
                return False
        return True

    def _pick_next_concept(self, user_id: str) -> str:
        self._ensure_user_mastery(user_id)
        available: list[tuple[str, float]] = []
        for cid, concept in self._concepts.items():
            if not self._prerequisites_met(user_id, concept):
                continue
            mastery = self._mastery[user_id][cid]
            available.append((cid, mastery.mastery_level))
        if not available:
            raise SocraticTutorError("No available concepts with prerequisites met.")
        available.sort(key=lambda x: x[1])
        return available[0][0]

    def _next_question_index(self, user_id: str, concept_id: str) -> int | None:
        streaks = self._question_streaks[user_id][concept_id]
        for idx, streak in enumerate(streaks):
            if streak < 2:
                return idx
        return None

    def ask_question(self, session_id: str, concept_id: str | None = None) -> dict[str, Any]:
        """Return the next question for the requested or auto-selected concept."""
        if session_id not in self._sessions:
            raise SocraticTutorError(f"Session {session_id!r} not found.")
        session = self._sessions[session_id]
        user_id = session.user_id
        self._ensure_user_mastery(user_id)

        if concept_id is None:
            concept_id = self._pick_next_concept(user_id)
        elif concept_id not in self._concepts:
            raise SocraticTutorError(f"Concept {concept_id!r} not found.")
        elif not self._prerequisites_met(user_id, self._concepts[concept_id]):
            raise SocraticTutorError(f"Prerequisites for concept {concept_id!r} are not met.")

        session.active_concept_id = concept_id
        concept = self._concepts[concept_id]
        q_idx = self._next_question_index(user_id, concept_id)
        if q_idx is None:
            return {
                "mastered": True,
                "concept_id": concept_id,
                "message": "All questions mastered. Consider moving to the next concept.",
            }

        question_data = concept["questions"][q_idx]
        self._session_question[session_id] = (concept_id, q_idx)
        result = {
            "concept_id": concept_id,
            "question_index": q_idx,
            "question": question_data["question"],
        }
        session.history.append(
            {
                "type": "question",
                "concept_id": concept_id,
                "question": question_data["question"],
                "timestamp": time.time(),
            }
        )
        return result

    # ------------------------------------------------------------------
    # Answer evaluation
    # ------------------------------------------------------------------

    def evaluate_answer(self, session_id: str, answer: str) -> dict[str, Any]:
        """Evaluate the user's answer and update mastery."""
        if session_id not in self._sessions:
            raise SocraticTutorError(f"Session {session_id!r} not found.")
        session = self._sessions[session_id]
        user_id = session.user_id
        if session_id not in self._session_question:
            raise SocraticTutorError("No active question for this session.")

        concept_id, q_idx = self._session_question[session_id]
        concept = self._concepts[concept_id]
        question_data = concept["questions"][q_idx]

        answer_lower = answer.lower()
        correct = any(
            pat.lower() in answer_lower for pat in question_data["expected_answer_patterns"]
        )

        self._ensure_user_mastery(user_id)
        mastery = self._mastery[user_id][concept_id]
        mastery.attempts += 1

        if correct:
            mastery.mastery_level = min(1.0, mastery.mastery_level + 0.15)
            mastery.correct_streak += 1
            self._question_streaks[user_id][concept_id][q_idx] += 1
            feedback = "Correct! Well done."
            mastery_delta = 0.15
            next_hint = None
        else:
            mastery.mastery_level = max(0.0, mastery.mastery_level - 0.05)
            mastery.correct_streak = 0
            self._question_streaks[user_id][concept_id][q_idx] = 0
            feedback = "That's not quite right."
            mastery_delta = -0.05
            hints = concept.get("hints", [])
            hint_index = question_data.get("hint_index", 0)
            next_hint = hints[hint_index] if hint_index < len(hints) else None

        session.history.append(
            {
                "type": "answer",
                "answer": answer,
                "correct": correct,
                "timestamp": time.time(),
            }
        )
        return {
            "correct": correct,
            "feedback": feedback,
            "mastery_delta": mastery_delta,
            "next_hint": next_hint,
        }

    # ------------------------------------------------------------------
    # Mastery & progress queries
    # ------------------------------------------------------------------

    def get_mastery(
        self, user_id: str, concept_id: str | None = None
    ) -> dict[str, Any] | list[dict[str, Any]]:
        """Return mastery for one concept or all concepts."""
        self._ensure_user_mastery(user_id)
        if concept_id is not None:
            if concept_id not in self._concepts:
                raise SocraticTutorError(f"Concept {concept_id!r} not found.")
            return asdict(self._mastery[user_id][concept_id])
        return [asdict(m) for m in self._mastery[user_id].values()]

    def get_progress_report(self, user_id: str) -> dict[str, Any]:
        """Overall stats: concepts started, mastered, and average mastery."""
        self._ensure_user_mastery(user_id)
        masteries = list(self._mastery[user_id].values())
        if not masteries:
            return {
                "concepts_started": 0,
                "concepts_mastered": 0,
                "avg_mastery": 0.0,
            }
        started = sum(1 for m in masteries if m.attempts > 0)
        mastered = sum(1 for m in masteries if m.mastery_level >= 0.8)
        avg = sum(m.mastery_level for m in masteries) / len(masteries)
        return {
            "concepts_started": started,
            "concepts_mastered": mastered,
            "avg_mastery": avg,
        }

    def reset_mastery(self, user_id: str, concept_id: str | None = None) -> None:
        """Reset mastery for one concept or all concepts."""
        self._ensure_user_mastery(user_id)
        if concept_id is not None:
            if concept_id not in self._concepts:
                raise SocraticTutorError(f"Concept {concept_id!r} not found.")
            concept = self._concepts[concept_id]
            self._mastery[user_id][concept_id] = ConceptMastery(
                concept_id=concept_id, name=concept["name"]
            )
            num_questions = len(concept.get("questions", []))
            self._question_streaks[user_id][concept_id] = [0] * num_questions
        else:
            for cid, concept in self._concepts.items():
                self._mastery[user_id][cid] = ConceptMastery(concept_id=cid, name=concept["name"])
                num_questions = len(concept.get("questions", []))
                self._question_streaks[user_id][cid] = [0] * num_questions


# ---------------------------------------------------------------------------
# Singleton & registry
# ---------------------------------------------------------------------------

DEFAULT_SOCRATIC_TUTOR = SocraticTutor()

SOCRATIC_TUTOR_REGISTRY: dict[str, SocraticTutor] = {
    "default": DEFAULT_SOCRATIC_TUTOR,
}
