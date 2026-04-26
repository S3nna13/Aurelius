"""Safe shell-command planner.

Given a natural-language intent and a model-agnostic ``generate_fn``,
produce a :class:`ShellPlan` whose commands are annotated with safety
risk scores. This module *plans* only: it NEVER executes commands.

Classification relies on a static allow-list (safe commands such as
``ls``, ``git``, ``rg``), a deny-list (forbidden base commands such as
``dd``, ``mkfs``), and a set of deny regex patterns matched against the
full command string (e.g. ``rm -rf /``, ``curl * | sh``, fork bombs).

Pure stdlib: only :mod:`shlex`, :mod:`re`, :mod:`json`, and
:mod:`dataclasses`. No silent fallbacks: malformed generator output is
reported via ``ShellPlan.warnings`` and yields an empty command list.
"""

from __future__ import annotations

import json
import re
import shlex
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

__all__ = [
    "ALLOWLIST",
    "DENYLIST",
    "DENY_PATTERNS",
    "RISK_ORDER",
    "ShellCommand",
    "ShellCommandPlanner",
    "ShellPlan",
]


#: Base commands considered generally safe (read-only or project-local).
ALLOWLIST: tuple[str, ...] = (
    "ls",
    "pwd",
    "echo",
    "cat",
    "head",
    "tail",
    "wc",
    "sort",
    "uniq",
    "cut",
    "tr",
    "diff",
    "git",
    "grep",
    "rg",
    "find",
    "fd",
    "uv",
    "ruff",
    "mypy",
    "black",
    "isort",
    "which",
    "whoami",
    "date",
    "true",
    "false",
    "stat",
    "file",
    "tree",
)

#: Commands that are frequently abused for arbitrary code execution.
#: They are NOT on the allowlist; any command whose base name matches
#: one of these is classified as *caution* (requires confirmation).
_CAUTION_COMMANDS: tuple[str, ...] = (
    "python",
    "python3",
    "pip",
    "pip3",
    "node",
    "npm",
    "npx",
    "yarn",
    "pnpm",
    "make",
    "cmake",
    "ninja",
    "env",
)

#: Base commands whose invocation is always at least "dangerous".
DENYLIST: tuple[str, ...] = (
    "dd",
    "mkfs",
    "mkswap",
    "fdisk",
    "parted",
    "shred",
    "wipefs",
    "shutdown",
    "reboot",
    "halt",
    "poweroff",
    "kexec",
    "umount",
    "chattr",
)

#: Regex patterns matched against the *full* command string. Any hit
#: marks the command as ``forbidden``.
#: Regex patterns matched against the *full* command string. Any hit
#: marks the command as ``forbidden``.
DENY_PATTERNS: tuple[tuple[str, str], ...] = (
    # rm with recursive + force against root/home — catch clustered and
    # separated flags (e.g. ``rm -r -f /``).
    (
        r"\brm\s+(-[a-zA-Z]*r[a-zA-Z]*f|-[a-zA-Z]*f[a-zA-Z]*r|-rf|-fr|--recursive\s+--force|--force\s+--recursive)\s+(/|~|\$HOME|/\*)",
        "rm -rf against root or home",
    ),
    (r"\brm\s+(-[a-zA-Z]*\s+)*-r\b.*\b-f\b\s+(/|~|\$HOME|/\*)", "rm -r -f against root or home"),
    (r"\brm\s+(-[a-zA-Z]*\s+)*-f\b.*\b-r\b\s+(/|~|\$HOME|/\*)", "rm -f -r against root or home"),
    (r"\bsudo\s+rm\b", "sudo rm"),
    (r"\bchmod\s+-?R?\s*777\b", "chmod 777 (world-writable)"),
    (r"\bchmod\s+-?R?\s*666\b", "chmod 666 (world-writable)"),
    # Pipe to shell — allow sudo in between (curl | sudo bash)
    (r"\bcurl\b[^|;&]*\|\s*(?:sudo\s+)?(sh|bash|zsh)\b", "curl | sh remote execution"),
    (r"\bwget\b[^|;&]*\|\s*(?:sudo\s+)?(sh|bash|zsh)\b", "wget | sh remote execution"),
    (r"\bdd\s+if=", "dd if= raw disk write"),
    (r"\bmkfs\.", "mkfs filesystem create"),
    (r">\s*/dev/(sd[a-z]|nvme|hd[a-z]|disk)", "redirect to raw disk device"),
    (r"(^|[\s;&|])eval\s", "shell eval of untrusted input"),
    (r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:", "fork bomb"),
    (r"\bchown\s+-R\s+.*\s+/\b", "recursive chown of root"),
    (r">\s*/etc/passwd\b", "overwrite /etc/passwd"),
    (r">\s*/etc/shadow\b", "overwrite /etc/shadow"),
    # Command chaining / substitution that destroys the safety model of
    # single-command classification.
    (r"[;&]", "command chaining with ; or &"),
    (r"\|\s*\S", "pipe to another command"),
    (r"\$\([^)]*\)", "command substitution $()"),
    (r"`[^`]*`", "command substitution backticks"),
)

#: Ordering used to combine per-command risks into an overall risk.
RISK_ORDER: tuple[str, ...] = ("safe", "caution", "dangerous", "forbidden")


def _risk_rank(risk: str) -> int:
    try:
        return RISK_ORDER.index(risk)
    except ValueError:
        return len(RISK_ORDER)  # unknown risks rank highest


@dataclass(frozen=True)
class ShellCommand:
    """A single shell command annotated with a risk classification."""

    cmd: str
    args: list[str]
    risk: str
    risk_reason: str
    requires_confirmation: bool


@dataclass(frozen=True)
class ShellPlan:
    """A plan produced by :class:`ShellCommandPlanner`.

    ``overall_risk`` is the max of the per-command risks (``safe`` if
    empty). ``warnings`` collects parse-time diagnostics.
    """

    commands: list[ShellCommand]
    overall_risk: str
    warnings: list[str] = field(default_factory=list)


class ShellCommandPlanner:
    """Plans shell commands from intents using a supplied generator.

    Parameters
    ----------
    generate_fn:
        Callable mapping an intent string to raw candidate command
        text. The text may be newline-separated raw commands, or a
        JSON array / object. No execution is ever performed.
    allowlist, denylist:
        Optional overrides; values are merged with the module
        defaults (never silently replace).
    extra_denypatterns:
        Optional extra regex patterns (string form) whose match marks
        a command as ``forbidden`` with a generic reason.
    """

    def __init__(
        self,
        generate_fn: Callable[[str], str],
        allowlist: list[str] | None = None,
        denylist: list[str] | None = None,
        extra_denypatterns: list[str] | None = None,
    ) -> None:
        if not callable(generate_fn):
            raise TypeError("generate_fn must be callable")
        self._generate_fn = generate_fn
        self._allowlist: frozenset[str] = frozenset(ALLOWLIST) | frozenset(allowlist or ())
        self._denylist: frozenset[str] = frozenset(DENYLIST) | frozenset(denylist or ())
        self._cautionlist: frozenset[str] = frozenset(_CAUTION_COMMANDS)
        patterns: list[tuple[re.Pattern[str], str]] = [
            (re.compile(pat), reason) for pat, reason in DENY_PATTERNS
        ]
        for pat in extra_denypatterns or ():
            patterns.append((re.compile(pat), f"custom deny pattern: {pat}"))
        self._deny_patterns: tuple[tuple[re.Pattern[str], str], ...] = tuple(patterns)

    # ------------------------------------------------------------------
    # public API
    # ------------------------------------------------------------------
    def classify_command(self, cmd: str) -> ShellCommand:
        """Classify a single shell command string (synchronous)."""
        if not isinstance(cmd, str):
            raise TypeError("cmd must be str")
        raw = cmd.strip()
        if not raw:
            return ShellCommand(
                cmd="",
                args=[],
                risk="caution",
                risk_reason="empty command",
                requires_confirmation=True,
            )

        # 1) Pattern scan on full string (forbidden wins immediately).
        for pattern, reason in self._deny_patterns:
            if pattern.search(raw):
                base, args = self._split(raw)
                return ShellCommand(
                    cmd=base,
                    args=args,
                    risk="forbidden",
                    risk_reason=reason,
                    requires_confirmation=True,
                )

        # 2) Tokenize for base-command lookup.
        base, args = self._split(raw)
        if not base:
            return ShellCommand(
                cmd="",
                args=args,
                risk="caution",
                risk_reason="unparseable command",
                requires_confirmation=True,
            )
        base_name = base.rsplit("/", 1)[-1]  # strip leading path

        if base_name in self._denylist:
            return ShellCommand(
                cmd=base,
                args=args,
                risk="dangerous",
                risk_reason=f"base command '{base_name}' is on deny-list",
                requires_confirmation=True,
            )

        if base_name in self._allowlist:
            return ShellCommand(
                cmd=base,
                args=args,
                risk="safe",
                risk_reason=f"base command '{base_name}' is on allow-list",
                requires_confirmation=False,
            )

        if base_name in self._cautionlist:
            return ShellCommand(
                cmd=base,
                args=args,
                risk="caution",
                risk_reason=f"base command '{base_name}' can execute arbitrary code",
                requires_confirmation=True,
            )

        return ShellCommand(
            cmd=base,
            args=args,
            risk="caution",
            risk_reason=f"base command '{base_name}' is unknown (neither allow nor deny)",
            requires_confirmation=True,
        )

    def plan(self, intent: str) -> ShellPlan:
        """Produce a :class:`ShellPlan` for the given natural-language intent."""
        if not isinstance(intent, str):
            raise TypeError("intent must be str")
        warnings: list[str] = []
        try:
            raw = self._generate_fn(intent)
        except Exception as exc:  # noqa: BLE001 -- surface generator failure as warning
            warnings.append(f"generate_fn raised {type(exc).__name__}: {exc}")
            return ShellPlan(commands=[], overall_risk="safe", warnings=warnings)

        if not isinstance(raw, str):
            warnings.append(
                f"generate_fn returned non-string ({type(raw).__name__}); refusing to plan"
            )
            return ShellPlan(commands=[], overall_risk="safe", warnings=warnings)

        if not raw.strip():
            return ShellPlan(commands=[], overall_risk="safe", warnings=warnings)

        candidates, parse_warnings = self._parse_candidates(raw)
        warnings.extend(parse_warnings)

        commands: list[ShellCommand] = []
        for text in candidates:
            text = text.strip()
            if not text:
                continue
            if text.startswith("#"):
                continue  # shell comment
            commands.append(self.classify_command(text))

        if commands:
            overall = max(commands, key=lambda c: _risk_rank(c.risk)).risk
        else:
            overall = "safe"
        return ShellPlan(commands=commands, overall_risk=overall, warnings=warnings)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    @staticmethod
    def _split(raw: str) -> tuple[str, list[str]]:
        """Tokenize a command using POSIX shell rules; never raise."""
        try:
            tokens = shlex.split(raw, posix=True)
        except ValueError:
            # Fall back to whitespace split so classification can still
            # proceed (unmatched quote, etc.).
            tokens = raw.split()
        if not tokens:
            return "", []
        return tokens[0], tokens[1:]

    @staticmethod
    def _parse_candidates(raw: str) -> tuple[list[str], list[str]]:
        """Return (candidates, warnings) parsed from generator output.

        Robust to three shapes:
        1. JSON array of strings.
        2. JSON object with a ``commands`` list.
        3. Newline-separated plain text.

        Mixed / malformed JSON that *looks* like JSON (leading ``[`` or
        ``{``) but fails to parse is reported as a warning and yields
        no candidates (no silent fallback to line-splitting in that
        case).
        """
        text = raw.strip()
        warnings: list[str] = []
        first = text[:1]

        if first in ("[", "{"):
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                warnings.append(f"candidate output looked like JSON but failed to parse: {exc.msg}")
                return [], warnings
            if isinstance(obj, list):
                out: list[str] = []
                for item in obj:
                    if isinstance(item, str):
                        out.append(item)
                    else:
                        warnings.append(
                            f"JSON array element is not a string: {type(item).__name__}"
                        )
                return out, warnings
            if isinstance(obj, dict):
                cmds = obj.get("commands")
                if isinstance(cmds, list) and all(isinstance(x, str) for x in cmds):
                    return list(cmds), warnings
                warnings.append("JSON object missing valid 'commands' list[str]")
                return [], warnings
            warnings.append(f"JSON root is {type(obj).__name__}, expected list or object")
            return [], warnings

        # Newline-separated plain text.
        return [line for line in text.splitlines()], warnings


def _assert_iterable_of_str(name: str, value: Iterable[str] | None) -> None:
    if value is None:
        return
    for x in value:
        if not isinstance(x, str):
            raise TypeError(f"{name} must contain only str, got {type(x).__name__}")
