"""Synthetic instruction-tuning data generators for Aurelius.

Implements three schemes from the literature:

* Magpie (Xu 2024, arXiv:2406.08464) — self-instruct by priming the
  model with chat header tokens, letting it produce a user query, and
  then sampling the assistant completion.
* Self-Instruct (Wang 2022, arXiv:2212.10560) — seed tasks + iterative
  expansion using the model as a bootstrapper.
* Evol-Instruct (Xu 2023, arXiv:2304.12244) — evolve seed instructions
  through a sequence of complexity-raising operators.

All generators accept a ``generate_fn: Callable[[str], str]`` so they
can be unit-tested with deterministic fakes; in production the caller
wires in a real LM decoder.

The output surface is ``InstructionSample`` — a small dataclass that
downstream SFT pipelines read. All generators sanitize outputs: any
sample whose instruction or response contains ChatML role-break tokens
is silently dropped (never raised), because a model that tries to
break role is a classic prompt-injection failure mode and such samples
would poison training.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from src.chat.chatml_template import (
    IM_END,
    IM_START,
    ChatMLTemplate,
    Message,
)

logger = logging.getLogger(__name__)

# Role-break markers we consider contaminant if they appear in the
# model's output for either the instruction or the response.
_ROLE_BREAK_TOKENS = (IM_START, IM_END)


@dataclass
class InstructionSample:
    """A single (instruction, response) pair produced by a generator.

    Attributes:
        instruction: The user-facing query / task.
        response: The assistant's answer to ``instruction``.
        source: Free-form provenance tag, e.g. ``"magpie"``,
            ``"self_instruct:iter=1"``, ``"evol_instruct:deepen"``.
        difficulty: Monotone scalar in [0, 1] that callers may use for
            curriculum sampling. Evol-Instruct populates this with the
            step index normalized by total steps; Magpie / Self-Instruct
            leave it at 0.0 unless overridden.
        tags: Free-form labels, e.g. the operator name for Evol, or
            ``["seed"]`` for the original seeds in Self-Instruct.
    """

    instruction: str
    response: str
    source: str
    difficulty: float = 0.0
    tags: list[str] = field(default_factory=list)


def _contains_role_break(text: str) -> bool:
    """Return True iff ``text`` embeds a ChatML role-break token."""
    if not isinstance(text, str):
        return True
    return any(tok in text for tok in _ROLE_BREAK_TOKENS)


def _safe_call(generate_fn: Callable[[str], str], prompt: str) -> str | None:
    """Invoke ``generate_fn`` but swallow exceptions.

    We log and return ``None`` rather than propagating — a synthetic
    data pipeline that aborts on the first bad decode would be fragile
    against transient LM failures (timeouts, OOMs, finite-context
    overflows). The caller treats ``None`` as "skip this sample".
    """
    try:
        out = generate_fn(prompt)
    except Exception as exc:  # noqa: BLE001 — intentional broad catch
        logger.warning("generate_fn raised %s: %s", type(exc).__name__, exc)
        return None
    if not isinstance(out, str):
        logger.warning("generate_fn returned non-str: %s", type(out).__name__)
        return None
    return out


# --------------------------------------------------------------- Magpie


class MagpieGenerator:
    """Magpie-style self-instruct generator.

    The Magpie recipe exploits the fact that a chat-SFT'd model, when
    primed only with the opening ``<|im_start|>user\\n`` header, will
    hallucinate a plausible user turn. We then close the turn and
    sample the assistant reply. Two forward passes per sample, no seed
    prompts, no curated taxonomy.
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        chat_template: str = "chatml",
    ) -> None:
        if not callable(generate_fn):
            raise TypeError("generate_fn must be callable: str -> str")
        if chat_template != "chatml":
            # We only wire up chatml right now; other templates would
            # need their own header-token priming strings. Fail loud.
            raise ValueError(
                f"MagpieGenerator: unsupported chat_template={chat_template!r} "
                f"(only 'chatml' is implemented)"
            )
        self.generate_fn = generate_fn
        self.chat_template = chat_template
        self._tpl = ChatMLTemplate()

    def _user_prime(self) -> str:
        # Bare opening header for the user turn, no content — the LM
        # fills in the instruction.
        return f"{IM_START}user\n"

    def _assistant_prime(self, instruction: str) -> str:
        # A complete user turn followed by an open assistant header.
        # We use the template encoder so that if the ChatML wire format
        # ever changes, Magpie follows.
        msgs = [Message(role="user", content=instruction)]
        return self._tpl.encode(msgs, add_generation_prompt=True)

    def generate(self, n: int = 10, seed: int = 0) -> list[InstructionSample]:
        """Generate up to ``n`` samples. Samples with contaminated
        outputs are dropped; the returned list may therefore be shorter
        than ``n``.
        """
        if n < 0:
            raise ValueError(f"n must be >= 0, got {n}")
        if n == 0:
            return []

        rng = random.Random(seed)
        out: list[InstructionSample] = []
        for i in range(n):
            # The RNG is threaded into the prompt so that fake
            # generate_fns can be deterministic per-sample. A real LM
            # would ignore the nonce.
            nonce = rng.random()
            user_prompt = f"{self._user_prime()}"
            raw_instr = _safe_call(self.generate_fn, f"{user_prompt}[magpie:user:{i}:{nonce}]")
            if raw_instr is None:
                continue
            instruction = raw_instr.strip()
            if not instruction or _contains_role_break(instruction):
                continue

            asst_prompt = self._assistant_prime(instruction)
            raw_resp = _safe_call(self.generate_fn, f"{asst_prompt}[magpie:assistant:{i}:{nonce}]")
            if raw_resp is None:
                continue
            response = raw_resp.strip()
            if not response or _contains_role_break(response):
                continue

            out.append(
                InstructionSample(
                    instruction=instruction,
                    response=response,
                    source="magpie",
                    difficulty=0.0,
                    tags=["magpie"],
                )
            )
        return out


# --------------------------------------------------------------- Self-Instruct


class SelfInstructGenerator:
    """Seed + bootstrap expansion (Wang 2022).

    Starts from a list of curated seed tasks and, for ``n_iterations``
    rounds, asks the LM to (a) write a new instruction analogous to a
    randomly sampled seed and (b) answer it. Each round's successful
    outputs are added back into the pool so later rounds compound.
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        seed_tasks: Sequence[str],
    ) -> None:
        if not callable(generate_fn):
            raise TypeError("generate_fn must be callable: str -> str")
        if not seed_tasks:
            raise ValueError("SelfInstructGenerator: seed_tasks must be non-empty")
        self.generate_fn = generate_fn
        self.seed_tasks: list[str] = list(seed_tasks)

    def _format_instruction_prompt(self, pool: Sequence[str]) -> str:
        examples = "\n".join(f"- {t}" for t in pool[-8:])
        return (
            "You are generating new instruction-tuning examples. "
            "Given these example instructions:\n"
            f"{examples}\n"
            "Write one new instruction in the same style. "
            "Output only the instruction text."
        )

    def _format_response_prompt(self, instruction: str) -> str:
        return (
            "Write a high-quality response to the following instruction.\n"
            f"Instruction: {instruction}\n"
            "Response:"
        )

    def expand(
        self,
        n_iterations: int = 3,
        per_iter: int = 5,
    ) -> list[InstructionSample]:
        if n_iterations < 0:
            raise ValueError(f"n_iterations must be >= 0, got {n_iterations}")
        if per_iter < 0:
            raise ValueError(f"per_iter must be >= 0, got {per_iter}")

        # Start pool with seeds (emitted as first-class samples so the
        # caller does not have to re-stitch).
        samples: list[InstructionSample] = []
        pool: list[str] = list(self.seed_tasks)
        for seed in self.seed_tasks:
            samples.append(
                InstructionSample(
                    instruction=seed,
                    response="",
                    source="self_instruct:seed",
                    difficulty=0.0,
                    tags=["seed"],
                )
            )

        for it in range(n_iterations):
            for _ in range(per_iter):
                instr_prompt = self._format_instruction_prompt(pool)
                raw_instr = _safe_call(self.generate_fn, instr_prompt)
                if raw_instr is None:
                    continue
                instruction = raw_instr.strip()
                if not instruction or _contains_role_break(instruction):
                    continue

                resp_prompt = self._format_response_prompt(instruction)
                raw_resp = _safe_call(self.generate_fn, resp_prompt)
                if raw_resp is None:
                    continue
                response = raw_resp.strip()
                if not response or _contains_role_break(response):
                    continue

                samples.append(
                    InstructionSample(
                        instruction=instruction,
                        response=response,
                        source=f"self_instruct:iter={it}",
                        difficulty=0.0,
                        tags=["self_instruct"],
                    )
                )
                pool.append(instruction)
        return samples


# --------------------------------------------------------------- Evol-Instruct


class EvolInstructGenerator:
    """Evol-Instruct complexity evolution (Xu 2023).

    Given a seed instruction, applies a sequence of operators that
    rewrite it into a harder/more specific variant, then samples a
    response for each step. Records the step-wise evolution so
    downstream curricula can sort by ``difficulty``.
    """

    OPERATORS = frozenset({"deepen", "concretize", "constrain", "reason", "complicate"})

    _OPERATOR_PROMPTS = {
        "deepen": (
            "Rewrite the instruction below so that it requires deeper "
            "domain knowledge to answer. Preserve the topic.\n"
            "Instruction: {instr}\nRewritten:"
        ),
        "concretize": (
            "Rewrite the instruction below by replacing any general "
            "concepts with more specific ones.\n"
            "Instruction: {instr}\nRewritten:"
        ),
        "constrain": (
            "Rewrite the instruction below by adding one concrete "
            "constraint (format, length, forbidden approach, etc).\n"
            "Instruction: {instr}\nRewritten:"
        ),
        "reason": (
            "Rewrite the instruction below so that solving it requires "
            "explicit multi-step reasoning.\n"
            "Instruction: {instr}\nRewritten:"
        ),
        "complicate": (
            "Rewrite the instruction below to be harder without "
            "changing its domain or intent. Do not make it ambiguous.\n"
            "Instruction: {instr}\nRewritten:"
        ),
    }

    def __init__(self, generate_fn: Callable[[str], str]) -> None:
        if not callable(generate_fn):
            raise TypeError("generate_fn must be callable: str -> str")
        self.generate_fn = generate_fn

    def _response_prompt(self, instruction: str) -> str:
        return f"Answer the following instruction.\nInstruction: {instruction}\nAnswer:"

    def evolve(
        self,
        seed: str,
        steps: int = 3,
        operator_sequence: list[str] | None = None,
    ) -> list[InstructionSample]:
        if not isinstance(seed, str) or not seed.strip():
            raise ValueError("seed must be a non-empty string")
        if steps < 0:
            raise ValueError(f"steps must be >= 0, got {steps}")

        if operator_sequence is not None:
            for op in operator_sequence:
                if op not in self.OPERATORS:
                    raise ValueError(
                        f"unknown evol operator {op!r}; "
                        f"valid operators are {sorted(self.OPERATORS)}"
                    )
            ops = list(operator_sequence)
        else:
            # Deterministic rotation through the canonical operator
            # list. We sort for reproducibility because frozenset
            # iteration order is implementation-defined.
            canonical = sorted(self.OPERATORS)
            ops = [canonical[i % len(canonical)] for i in range(steps)]

        # When the caller passes an explicit operator_sequence we
        # honour its length; ``steps`` is ignored in that case, which
        # is the behaviour the WizardLM paper uses in practice.
        if operator_sequence is None:
            ops = ops[:steps]

        samples: list[InstructionSample] = []
        current = seed.strip()
        total = max(len(ops), 1)
        for idx, op in enumerate(ops):
            prompt = self._OPERATOR_PROMPTS[op].format(instr=current)
            raw_evolved = _safe_call(self.generate_fn, prompt)
            if raw_evolved is None:
                # Skip this step but keep going from the previous
                # instruction so the chain does not collapse.
                continue
            evolved = raw_evolved.strip()
            if not evolved or _contains_role_break(evolved):
                continue

            raw_resp = _safe_call(self.generate_fn, self._response_prompt(evolved))
            if raw_resp is None:
                current = evolved
                continue
            response = raw_resp.strip()
            if not response or _contains_role_break(response):
                current = evolved
                continue

            samples.append(
                InstructionSample(
                    instruction=evolved,
                    response=response,
                    source=f"evol_instruct:{op}",
                    difficulty=(idx + 1) / total,
                    tags=[op, "evol_instruct"],
                )
            )
            current = evolved
        return samples


__all__ = [
    "InstructionSample",
    "MagpieGenerator",
    "SelfInstructGenerator",
    "EvolInstructGenerator",
]
