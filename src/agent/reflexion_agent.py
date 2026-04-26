"""Reflexion verbal RL agent for Aurelius.

Implements the Reflexion paradigm from Shinn et al. 2023
("Reflexion: Language Agents with Verbal Reinforcement Learning",
arXiv:2303.11366).

The core idea is simple: when an attempt fails, the agent generates a
*verbal* reflection that articulates what went wrong, and that reflection is
prepended to the next attempt's context.  Over successive attempts the agent
accumulates a sliding-window episodic memory of (attempt, outcome, reflection)
triples that serves as the verbal equivalent of a gradient signal.

Design choices
--------------
* Pure Python standard library: ``dataclasses``, ``typing``, ``collections``.
  No torch, no external ML libraries.
* A ``ReflexionMemory`` class provides the bounded sliding window and the
  text-context builder.
* ``ReflexionAgent.run`` executes the loop; it receives three callables from
  the caller so that the agent itself is model-agnostic.
* The agent is registered in ``AGENT_LOOP_REGISTRY["reflexion"]``.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class ReflexionConfig:
    """Hyper-parameters for a :class:`ReflexionAgent` run.

    Attributes
    ----------
    max_attempts:
        Maximum number of attempts to make before stopping regardless of
        outcome.  Minimum 1.
    max_reflection_tokens:
        Soft upper bound (in characters) on the reflection string generated
        per failed attempt.  Passed to ``reflect_fn`` as context but not
        enforced by the agent itself — enforcement is the responsibility of the
        underlying ``reflect_fn``.
    reflection_decay:
        Conceptual decay factor (0 < decay <= 1.0) for older reflections when
        building context.  In the current text-based implementation all stored
        reflections are included verbatim; the attribute is preserved for
        downstream use by callers that wish to implement weighted retrieval.
    early_stop_on_success:
        If ``True``, stop as soon as an attempt reaches ``success_threshold``.
    memory_size:
        Maximum number of reflections to retain in :class:`ReflexionMemory`.
        Oldest entries are evicted when the window overflows.
    """

    max_attempts: int = 4
    max_reflection_tokens: int = 256
    reflection_decay: float = 0.9
    early_stop_on_success: bool = True
    memory_size: int = 10


# ---------------------------------------------------------------------------
# Result datatypes
# ---------------------------------------------------------------------------


@dataclass
class ReflexionAttempt:
    """A single attempt recorded during :meth:`ReflexionAgent.run`.

    Attributes
    ----------
    attempt_idx:
        Zero-based index of this attempt within the run.
    output:
        Raw text produced by ``attempt_fn`` for this attempt.
    score:
        Evaluation score in ``[0.0, 1.0]`` as returned by ``eval_fn``.
    reflection:
        Verbal reflection generated after a failed attempt.  Empty string
        when the attempt succeeded or no reflection was generated.
    tokens_used:
        Character length of ``output`` (a lightweight proxy for token count
        in this implementation; callers can override if a real tokeniser is
        available).
    """

    attempt_idx: int
    output: str
    score: float
    reflection: str
    tokens_used: int


@dataclass
class ReflexionResult:
    """Complete result of a :meth:`ReflexionAgent.run` invocation.

    Attributes
    ----------
    attempts:
        Ordered list of every :class:`ReflexionAttempt` made.
    final_output:
        The output text of the last attempt executed.
    success:
        ``True`` if any attempt reached ``success_threshold``.
    n_attempts:
        Total number of attempts made (``len(attempts)``).
    best_score:
        Highest ``score`` seen across all attempts.
    reflection_history:
        Ordered list of non-empty reflection strings generated during the run.
    """

    attempts: list[ReflexionAttempt] = field(default_factory=list)
    final_output: str = ""
    success: bool = False
    n_attempts: int = 0
    best_score: float = 0.0
    reflection_history: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Memory
# ---------------------------------------------------------------------------


class ReflexionMemory:
    """Sliding-window episodic memory for verbal reflections.

    Parameters
    ----------
    max_size:
        Maximum number of reflections to retain.  When a new reflection
        would exceed this limit, the oldest entry is evicted first
        (FIFO order).
    """

    def __init__(self, max_size: int) -> None:
        if max_size < 1:
            raise ValueError("max_size must be >= 1")
        self._max_size: int = max_size
        self._store: deque[str] = deque()

    # ------------------------------------------------------------------
    # Mutation
    # ------------------------------------------------------------------

    def add(self, reflection: str) -> None:
        """Append *reflection* to memory, evicting the oldest if at capacity."""
        if len(self._store) >= self._max_size:
            self._store.popleft()
        self._store.append(reflection)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def get_all(self) -> list[str]:
        """Return all stored reflections in insertion order (oldest first)."""
        return list(self._store)

    def build_context(self, decay: float = 0.9) -> str:  # noqa: ARG002
        """Render stored reflections as a text block for inclusion in context.

        Each reflection is prefixed with ``[Reflection N]:`` (1-based).  The
        ``decay`` parameter is accepted for API compatibility but text output
        is not weighted — all reflections are included verbatim.  Callers that
        require score-weighted retrieval should process ``get_all()`` directly.

        Returns an empty string when no reflections are stored.
        """
        items = self.get_all()
        if not items:
            return ""
        lines = [f"[Reflection {i + 1}]: {r}" for i, r in enumerate(items)]
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Dunder
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._store)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class ReflexionAgent:
    """Reflexion verbal-RL agent.

    The agent does not own a model.  Instead it is parameterised at *call
    time* by three callables:

    * ``attempt_fn(context: str) -> str``
      Generates an attempt given the accumulated context (task description
      plus any reflection memory prepended by the agent).
    * ``eval_fn(output: str) -> float``
      Evaluates the attempt and returns a score in ``[0.0, 1.0]``.  A score
      of ``1.0`` is treated as a perfect success.
    * ``reflect_fn(prompt: str) -> str``
      Given a reflection prompt constructed by the agent, returns a verbal
      explanation of what went wrong.

    Parameters
    ----------
    config:
        :class:`ReflexionConfig` instance controlling loop behaviour.
    """

    def __init__(self, config: ReflexionConfig | None = None) -> None:
        self.config: ReflexionConfig = config if config is not None else ReflexionConfig()
        self._memory: ReflexionMemory = ReflexionMemory(self.config.memory_size)

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    def reflect(
        self,
        attempt_output: str,
        task_description: str,
        score: float,
        generate_fn: Callable[[str], str],
    ) -> str:
        """Generate a verbal reflection on a failed attempt.

        Constructs a structured prompt containing the task, the attempt
        output, and the score, then delegates to ``generate_fn``.

        Parameters
        ----------
        attempt_output:
            The text produced by the agent on this attempt.
        task_description:
            Original task given to the agent.
        score:
            Evaluation score for this attempt.
        generate_fn:
            A ``str -> str`` callable used to produce the reflection text.

        Returns
        -------
        str
            The verbal reflection returned by ``generate_fn``.
        """
        prompt = f"Task: {task_description}\nAttempt: {attempt_output}\nScore: {score}\nReflection:"
        return generate_fn(prompt)

    def run(
        self,
        task_description: str,
        attempt_fn: Callable[[str], str],
        score_fn: Callable[[str], float],
        reflect_fn: Callable[[str], str],
        success_threshold: float = 1.0,
    ) -> ReflexionResult:
        """Execute the Reflexion loop.

        For each attempt the agent:

        1. Builds a context string from the task and accumulated reflections.
        2. Calls ``attempt_fn(context)`` to generate an output.
        3. Calls ``score_fn(output)`` to score the output.
        4. If the score reaches ``success_threshold`` and
           ``config.early_stop_on_success`` is ``True``, stops.
        5. Otherwise, generates a verbal reflection via :meth:`reflect` and
           stores it in ``_memory`` for the next iteration.

        A fresh ``ReflexionMemory`` is used for each call to ``run`` so that
        multiple calls do not bleed state into one another.

        Parameters
        ----------
        task_description:
            Plain-text description of the task to solve.
        attempt_fn:
            ``context: str -> output: str`` callable.
        score_fn:
            ``output: str -> score: float`` callable.  Score must be in
            ``[0.0, 1.0]``.
        reflect_fn:
            ``prompt: str -> reflection: str`` callable.
        success_threshold:
            Score at or above which an attempt is considered successful.

        Returns
        -------
        ReflexionResult
            Full record of all attempts and their outcomes.
        """
        # Reset per-run memory so successive calls are independent.
        memory = ReflexionMemory(self.config.memory_size)
        attempts: list[ReflexionAttempt] = []
        reflection_history: list[str] = []
        success = False
        best_score = 0.0

        for attempt_idx in range(self.config.max_attempts):
            # Build context: task + any stored reflections.
            context_parts: list[str] = [f"Task: {task_description}"]
            reflection_context = memory.build_context(self.config.reflection_decay)
            if reflection_context:
                context_parts.append(reflection_context)
            context = "\n".join(context_parts)

            # Attempt.
            output = attempt_fn(context)
            score = score_fn(output)
            tokens_used = len(output)  # character-length proxy

            # Determine whether this attempt succeeded.
            attempt_succeeded = score >= success_threshold
            if score > best_score:
                best_score = score

            # Generate reflection only on failure.
            reflection = ""
            if not attempt_succeeded:
                reflection = self.reflect(
                    attempt_output=output,
                    task_description=task_description,
                    score=score,
                    generate_fn=reflect_fn,
                )
                memory.add(reflection)
                reflection_history.append(reflection)

            attempts.append(
                ReflexionAttempt(
                    attempt_idx=attempt_idx,
                    output=output,
                    score=score,
                    reflection=reflection,
                    tokens_used=tokens_used,
                )
            )

            if attempt_succeeded and self.config.early_stop_on_success:
                success = True
                break

        final_output = attempts[-1].output if attempts else ""
        if not success and best_score >= success_threshold:
            success = True

        return ReflexionResult(
            attempts=attempts,
            final_output=final_output,
            success=success,
            n_attempts=len(attempts),
            best_score=best_score,
            reflection_history=reflection_history,
        )

    # ------------------------------------------------------------------
    # Post-run helpers
    # ------------------------------------------------------------------

    def best_attempt(self, result: ReflexionResult) -> ReflexionAttempt:
        """Return the :class:`ReflexionAttempt` with the highest score.

        If multiple attempts share the maximum score, the last one is
        returned (latest improvement wins).

        Raises
        ------
        ValueError
            If ``result.attempts`` is empty.
        """
        if not result.attempts:
            raise ValueError("result has no attempts")
        return max(result.attempts, key=lambda a: (a.score, a.attempt_idx))

    def statistics(self, result: ReflexionResult) -> dict[str, float]:
        """Compute summary statistics for a completed run.

        Returns
        -------
        dict with keys:
            * ``success_rate`` — 1.0 if ``result.success`` else 0.0
            * ``n_attempts`` — total number of attempts
            * ``best_score`` — highest score seen
            * ``mean_score`` — arithmetic mean of all attempt scores
            * ``improvement`` — last attempt score minus first attempt score
        """
        if not result.attempts:
            return {
                "success_rate": 0.0,
                "n_attempts": 0.0,
                "best_score": 0.0,
                "mean_score": 0.0,
                "improvement": 0.0,
            }
        scores = [a.score for a in result.attempts]
        return {
            "success_rate": 1.0 if result.success else 0.0,
            "n_attempts": float(result.n_attempts),
            "best_score": result.best_score,
            "mean_score": sum(scores) / len(scores),
            "improvement": scores[-1] - scores[0],
        }


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

from src.agent import AGENT_LOOP_REGISTRY  # noqa: E402

AGENT_LOOP_REGISTRY["reflexion"] = ReflexionAgent


__all__ = [
    "ReflexionAgent",
    "ReflexionAttempt",
    "ReflexionConfig",
    "ReflexionMemory",
    "ReflexionResult",
]
