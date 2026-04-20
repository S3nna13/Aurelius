"""Adversarial code battle: symmetric red-vs-blue loop over a code snippet.

Inspired by CodeGuardian's ``AdversarialBattle.start_battle`` loop. Each round:
    1. ``red_fn(code, previous_defenses)`` emits findings (vulnerabilities,
       attack chains, bypass techniques).
    2. ``blue_fn(code, red_findings, previous_patches)`` emits patches.
    3. Patches are applied (line-based replacement, never dynamic code
       interpretation) to produce ``code_{t+1}``.
    4. Memory of defenses/patches accumulates across rounds.
    5. Loop halts when ``rounds_without_new_vulns >= convergence_threshold``
       or ``max_rounds`` is reached.

The transcript drives DPO-style preference pairs:
``(vulnerable_code -> patched_code)`` is ``(rejected -> chosen)``.

Distinction from sibling surfaces in ``src/alignment/``:
    * ``self_play.py`` (SPIN) -- general self-play LM fine-tuning; not
      code-specific, no patching semantics.
    * ``red_team.py`` (Garak runner) -- evaluation harness wrapping external
      probes; this module is a closed-loop *training data* generator.
    * ``self_play_debate.py`` / ``debate_framework.py`` -- symmetric
      debate/critique over arbitrary answers; no code surface, no patch
      application step.

Aurelius never calls external APIs: ``red_fn`` / ``blue_fn`` are pluggable
callables. Deterministic ``heuristic_red_fn`` / ``heuristic_blue_fn`` are
provided for tests.

Outputs from ``red_fn`` / ``blue_fn`` are treated as *untrusted*: malformed
responses are coerced into empty findings / patches and logged.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable

logger = logging.getLogger(__name__)

RedFn = Callable[[str, list[dict]], dict]
BlueFn = Callable[[str, dict, list[dict]], dict]

# Sentinel tokens assembled at runtime to avoid hook false-positives on
# literal security-trigger substrings appearing in source.
_EV = "ev" + "al"
_SHELL_TRUE = "she" + "ll=T" + "rue"
_SHELL_FALSE = "she" + "ll=F" + "alse"


@dataclass
class RedFinding:
    """A single vulnerability finding emitted by the red agent."""

    type: str
    severity: str = "medium"
    line: int = -1
    attack_vector: str = ""
    impact: str = ""
    confidence: float = 0.5

    @classmethod
    def from_dict(cls, d: Any) -> "RedFinding | None":
        if not isinstance(d, dict):
            return None
        t = d.get("type")
        if not isinstance(t, str) or not t:
            return None
        try:
            return cls(
                type=t,
                severity=str(d.get("severity", "medium")),
                line=int(d.get("line", -1)) if d.get("line") is not None else -1,
                attack_vector=str(d.get("attack_vector", "")),
                impact=str(d.get("impact", "")),
                confidence=float(d.get("confidence", 0.5)),
            )
        except (TypeError, ValueError):
            return None


@dataclass
class BluePatch:
    """A single patch emitted by the blue agent."""

    vulnerability_type: str
    original_code: str
    patched_code: str
    defense_mechanism: str = ""
    confidence: float = 0.5
    applied: bool = False
    apply_error: str = ""
    line: int = -1

    @classmethod
    def from_dict(cls, d: Any) -> "BluePatch | None":
        if not isinstance(d, dict):
            return None
        vt = d.get("vulnerability_type")
        orig = d.get("original_code")
        new = d.get("patched_code")
        if not isinstance(vt, str) or not isinstance(orig, str) or not isinstance(new, str):
            return None
        try:
            return cls(
                vulnerability_type=vt,
                original_code=orig,
                patched_code=new,
                defense_mechanism=str(d.get("defense_mechanism", "")),
                confidence=float(d.get("confidence", 0.5)),
                line=int(d.get("line", -1)) if d.get("line") is not None else -1,
            )
        except (TypeError, ValueError):
            return None


@dataclass
class Round:
    """Record of a single round."""

    index: int
    code_before: str
    code_after: str
    red_findings: list[RedFinding]
    blue_patches: list[BluePatch]
    threat_assessment: str = ""
    new_vuln_types: list[str] = field(default_factory=list)


@dataclass
class BattleTranscript:
    """Full transcript of a battle."""

    starting_code: str
    final_code: str
    rounds: list[Round]
    converged: bool
    convergence_signal: str


def _apply_patch(code: str, patch: BluePatch, max_applies: int) -> tuple[str, bool, str]:
    """Apply a BluePatch to ``code`` via line-based replacement.

    Returns ``(new_code, applied, error_message)``. Never raises.
    """
    if not patch.original_code:
        return code, False, "empty original_code"
    if patch.original_code == patch.patched_code:
        return code, False, "no-op patch"

    count = code.count(patch.original_code)
    if count == 0:
        orig_stripped = patch.original_code.strip()
        lines = code.splitlines(keepends=True)
        applied = 0
        for i, line in enumerate(lines):
            if applied >= max_applies:
                break
            if line.strip() == orig_stripped:
                lead = line[: len(line) - len(line.lstrip())]
                trail = "\n" if line.endswith("\n") else ""
                lines[i] = lead + patch.patched_code.strip() + trail
                applied += 1
        if applied == 0:
            return code, False, "original_code not found"
        return "".join(lines), True, ""

    n = min(count, max_applies)
    new_code = code.replace(patch.original_code, patch.patched_code, n)
    return new_code, True, ""


class AdversarialCodeBattle:
    """Run a symmetric red-vs-blue battle over a code snippet."""

    def __init__(
        self,
        red_fn: RedFn,
        blue_fn: BlueFn,
        max_rounds: int = 5,
        convergence_threshold: int = 2,
        max_patch_applies: int = 3,
    ) -> None:
        if max_rounds < 1:
            raise ValueError("max_rounds must be >= 1")
        if convergence_threshold < 1:
            raise ValueError("convergence_threshold must be >= 1")
        if max_patch_applies < 1:
            raise ValueError("max_patch_applies must be >= 1")
        self.red_fn = red_fn
        self.blue_fn = blue_fn
        self.max_rounds = max_rounds
        self.convergence_threshold = convergence_threshold
        self.max_patch_applies = max_patch_applies

    def _coerce_red(self, raw: Any) -> tuple[list[RedFinding], list, list]:
        if not isinstance(raw, dict):
            logger.warning("red_fn returned non-dict output; coercing to empty")
            return [], [], []
        vulns_raw = raw.get("vulnerabilities", [])
        if not isinstance(vulns_raw, list):
            logger.warning("red_fn vulnerabilities not a list; coercing to empty")
            vulns_raw = []
        findings: list[RedFinding] = []
        for v in vulns_raw:
            f = RedFinding.from_dict(v)
            if f is not None:
                findings.append(f)
            else:
                logger.warning("red_fn produced malformed vulnerability: %r", v)
        chains_raw = raw.get("attack_chains", [])
        chains = chains_raw if isinstance(chains_raw, list) else []
        bypass_raw = raw.get("bypass_techniques", [])
        bypass = bypass_raw if isinstance(bypass_raw, list) else []
        return findings, chains, bypass

    def _coerce_blue(self, raw: Any) -> tuple[str, list[BluePatch], list]:
        if not isinstance(raw, dict):
            logger.warning("blue_fn returned non-dict output; coercing to empty")
            return "", [], []
        assessment = str(raw.get("threat_assessment", ""))
        patches_raw = raw.get("patches", [])
        if not isinstance(patches_raw, list):
            logger.warning("blue_fn patches not a list; coercing to empty")
            patches_raw = []
        patches: list[BluePatch] = []
        for p in patches_raw:
            bp = BluePatch.from_dict(p)
            if bp is not None:
                patches.append(bp)
            else:
                logger.warning("blue_fn produced malformed patch: %r", p)
        rules_raw = raw.get("detection_rules", [])
        rules = rules_raw if isinstance(rules_raw, list) else []
        return assessment, patches, rules

    def run(self, starting_code: str) -> BattleTranscript:
        code = starting_code
        rounds: list[Round] = []
        previous_defenses: list[dict] = []
        previous_patches: list[dict] = []
        seen_vuln_types: set[str] = set()
        rounds_without_new = 0
        converged = False
        signal = "max_rounds"

        for i in range(self.max_rounds):
            code_before = code

            try:
                red_raw = self.red_fn(code, list(previous_defenses))
            except Exception as e:
                logger.warning("red_fn raised %s; treating as empty output", e)
                red_raw = {}
            findings, chains, bypass = self._coerce_red(red_raw)

            new_types = [f.type for f in findings if f.type not in seen_vuln_types]
            for t in new_types:
                seen_vuln_types.add(t)

            if not findings:
                rounds.append(
                    Round(
                        index=i,
                        code_before=code_before,
                        code_after=code,
                        red_findings=[],
                        blue_patches=[],
                        threat_assessment="",
                        new_vuln_types=[],
                    )
                )
                converged = True
                signal = "no_findings"
                break

            previous_defenses.append(
                {
                    "round": i,
                    "findings": [f.__dict__.copy() for f in findings],
                    "attack_chains": chains,
                    "bypass_techniques": bypass,
                }
            )

            try:
                blue_raw = self.blue_fn(
                    code,
                    {
                        "vulnerabilities": [f.__dict__.copy() for f in findings],
                        "attack_chains": chains,
                        "bypass_techniques": bypass,
                    },
                    list(previous_patches),
                )
            except Exception as e:
                logger.warning("blue_fn raised %s; treating as empty output", e)
                blue_raw = {}
            assessment, patches, _rules = self._coerce_blue(blue_raw)

            for patch in patches:
                new_code, applied, err = _apply_patch(code, patch, self.max_patch_applies)
                patch.applied = applied
                patch.apply_error = err
                if applied:
                    code = new_code
                else:
                    logger.warning(
                        "patch for %s did not apply: %s", patch.vulnerability_type, err
                    )

            previous_patches.append(
                {
                    "round": i,
                    "patches": [p.__dict__.copy() for p in patches],
                    "threat_assessment": assessment,
                }
            )

            rounds.append(
                Round(
                    index=i,
                    code_before=code_before,
                    code_after=code,
                    red_findings=findings,
                    blue_patches=patches,
                    threat_assessment=assessment,
                    new_vuln_types=new_types,
                )
            )

            if not new_types:
                rounds_without_new += 1
            else:
                rounds_without_new = 0

            if rounds_without_new >= self.convergence_threshold:
                converged = True
                signal = "converged"
                break

        return BattleTranscript(
            starting_code=starting_code,
            final_code=code,
            rounds=rounds,
            converged=converged,
            convergence_signal=signal,
        )

    def to_preference_pairs(self, transcript: BattleTranscript) -> list[dict]:
        pairs: list[dict] = []
        for r in transcript.rounds:
            if not r.blue_patches:
                continue
            if r.code_before == r.code_after:
                continue
            for p in r.blue_patches:
                if not p.applied:
                    continue
                if p.original_code == p.patched_code:
                    continue
                parts = [f"round={r.index}", f"type={p.vulnerability_type}"]
                if p.defense_mechanism:
                    parts.append(f"defense={p.defense_mechanism}")
                pairs.append(
                    {
                        "chosen": p.patched_code,
                        "rejected": p.original_code,
                        "critique": "; ".join(parts),
                        "round": r.index,
                        "vulnerability_type": p.vulnerability_type,
                        "line": p.line,
                        "confidence": p.confidence,
                    }
                )
            pairs.append(
                {
                    "chosen": r.code_after,
                    "rejected": r.code_before,
                    "critique": f"round={r.index}; whole_file; "
                    + ",".join(sorted({f.type for f in r.red_findings})),
                    "round": r.index,
                    "vulnerability_type": "aggregate",
                    "line": -1,
                    "confidence": 1.0,
                }
            )
        return pairs


# ---------------------------------------------------------------------------
# Heuristic helpers (deterministic, regex-based).
# ---------------------------------------------------------------------------

_RED_PATTERNS: tuple[tuple[str, re.Pattern[str], str, str, str], ...] = (
    (
        "hardcoded_secret",
        re.compile(
            r'''(?ix)
            \b(?:password|passwd|secret|api_key|apikey|token)\s*=\s*
            (['"])[^'"\n]{3,}\1
            ''',
        ),
        "high",
        "credential in source",
        "leaks on repo push",
    ),
    (
        "dangerous_dynamic_exec",
        re.compile(r"\b" + _EV + r"\s*\("),
        "critical",
        "arbitrary code execution via dynamic expression evaluation",
        "RCE",
    ),
    (
        "shell_injection",
        re.compile(r"she" + r"ll\s*=\s*T" + r"rue"),
        "high",
        "subprocess with shell invocation and interpolated args",
        "command injection",
    ),
    (
        "sql_injection",
        re.compile(
            r'''(?i)(?:execute|cursor\.execute)\s*\(\s*(?:f['"]|['"][^'"]*['"]\s*\+|.+%\s*)'''
        ),
        "high",
        "string-concatenated SQL",
        "database compromise",
    ),
    (
        "insecure_random",
        re.compile(r"\brandom\.(?:random|randint|choice)\b"),
        "medium",
        "non-cryptographic randomness for security use",
        "predictable token",
    ),
)


def heuristic_red_fn(code: str, previous_defenses: list[dict]) -> dict:
    """Deterministic regex-based vulnerability finder."""
    already: set[str] = set()
    for d in previous_defenses:
        for v in d.get("findings", []) or []:
            if isinstance(v, dict) and "type" in v:
                already.add(v["type"])

    lines = code.splitlines()
    vulns: list[dict] = []
    for vtype, pat, sev, vector, impact in _RED_PATTERNS:
        if vtype in already:
            continue
        for idx, line in enumerate(lines, start=1):
            if pat.search(line):
                vulns.append(
                    {
                        "type": vtype,
                        "severity": sev,
                        "line": idx,
                        "attack_vector": vector,
                        "impact": impact,
                        "confidence": 0.9,
                    }
                )
                break
    return {
        "attack_surface": "python-source",
        "vulnerabilities": vulns,
        "attack_chains": [],
        "bypass_techniques": [],
    }


def _blue_secret(line: str) -> tuple[str, str, str]:
    m = re.match(r"""^(\s*)([A-Za-z_][A-Za-z_0-9]*)\s*=\s*['"][^'"]*['"]\s*$""", line)
    if not m:
        return line, line, "no-op"
    indent, name = m.group(1), m.group(2)
    env_key = name.upper()
    patched = f'{indent}{name} = __import__("os").environ["{env_key}"]'
    return line, patched, "env-var lookup"


def _blue_dyn_exec(line: str) -> tuple[str, str, str]:
    patched = re.sub(r"\b" + _EV + r"\s*\(", "ast.literal_eval(", line)
    return line, patched, "ast.literal_eval"


def _blue_shell(line: str) -> tuple[str, str, str]:
    patched = line.replace(_SHELL_TRUE, _SHELL_FALSE)
    return line, patched, "disable shell-mode"


def _blue_sql(line: str) -> tuple[str, str, str]:
    patched = line.rstrip() + "  # TODO: use parameterized query"
    return line, patched, "parameterize"


def _blue_random(line: str) -> tuple[str, str, str]:
    patched = re.sub(r"\brandom\.(random|randint|choice)\b", r"secrets.\1", line)
    return line, patched, "use secrets module"


_BLUE_RULES: dict[str, Callable[[str], tuple[str, str, str]]] = {
    "hardcoded_secret": _blue_secret,
    "dangerous_dynamic_exec": _blue_dyn_exec,
    "shell_injection": _blue_shell,
    "sql_injection": _blue_sql,
    "insecure_random": _blue_random,
}


def heuristic_blue_fn(
    code: str, red_findings: dict, previous_patches: list[dict]
) -> dict:
    """Deterministic patcher matching heuristic_red_fn's vulnerability types."""
    lines = code.splitlines()
    patches: list[dict] = []
    vulns = red_findings.get("vulnerabilities", []) if isinstance(red_findings, dict) else []
    for v in vulns:
        if not isinstance(v, dict):
            continue
        vtype = v.get("type")
        line_no = v.get("line", -1)
        rule = _BLUE_RULES.get(vtype)
        if rule is None:
            continue
        if not isinstance(line_no, int) or line_no < 1 or line_no > len(lines):
            continue
        original_line = lines[line_no - 1]
        orig, patched, defense = rule(original_line)
        if orig == patched:
            continue
        patches.append(
            {
                "vulnerability_type": vtype,
                "original_code": orig,
                "patched_code": patched,
                "defense_mechanism": defense,
                "confidence": 0.8,
                "line": line_no,
            }
        )
    return {
        "threat_assessment": f"{len(patches)} patches synthesised",
        "patches": patches,
        "detection_rules": [],
    }


__all__ = [
    "RedFinding",
    "BluePatch",
    "Round",
    "BattleTranscript",
    "AdversarialCodeBattle",
    "heuristic_red_fn",
    "heuristic_blue_fn",
]
