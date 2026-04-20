"""Pre-tool-execution policy denylist (Tracecat-inspired).

This module provides a *static policy* guard for tool invocations. Given
a tool name and its arguments, it inspects the flattened argument text
against a set of regex denylist rules. If any rule matches, the verdict
is ``allowed=False`` (in strict mode) with the offending rules reported.

This is **not** an OS-level sandbox. Aurelius already has
``code_execution_sandbox.py`` for a stdout/stderr-capturing runtime. This
module is a pure policy layer: it vetoes calls before dispatch and never
runs anything -- only regex pattern matching is performed.

Pure stdlib: only :mod:`re`, :mod:`dataclasses`, :mod:`enum`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Iterable

__all__ = [
    "DEFAULT_DENYLIST",
    "DenylistCategory",
    "DenylistRule",
    "DenyVerdict",
    "ToolSandboxDenylist",
]


class DenylistCategory(str, Enum):
    """Top-level danger categories for denylist rules."""

    KERNEL_SURFACE = "kernel_surface"
    DESTRUCTIVE_FS = "destructive_fs"
    NETWORK_EXFIL = "network_exfil"
    PRIVILEGE_ESCALATION = "privilege_escalation"
    SENSITIVE_READS = "sensitive_reads"
    CODE_EXEC_PRIMITIVES = "code_exec_primitives"
    RESOURCE_EXHAUSTION = "resource_exhaustion"


@dataclass(frozen=True)
class DenylistRule:
    """A single regex-based denylist rule."""

    id: str
    category: DenylistCategory
    pattern: str
    message: str
    severity: str = "high"
    allow_override: bool = False


@dataclass(frozen=True)
class DenyVerdict:
    """Outcome of a denylist evaluation."""

    allowed: bool
    violated_rules: tuple[DenylistRule, ...] = field(default_factory=tuple)
    advice: str = ""


DEFAULT_DENYLIST: tuple[DenylistRule, ...] = (
    DenylistRule(
        id="kern.add_key",
        category=DenylistCategory.KERNEL_SURFACE,
        pattern=r"\badd_key\b",
        message="kernel keyring manipulation (add_key) is disallowed",
    ),
    DenylistRule(
        id="kern.bpf",
        category=DenylistCategory.KERNEL_SURFACE,
        pattern=r"\bbpf\s*\(",
        message="bpf() syscall is disallowed",
    ),
    DenylistRule(
        id="kern.mount",
        category=DenylistCategory.KERNEL_SURFACE,
        pattern=r"(?<![\w./-])mount\b(?![\w.-])",
        message="mount is disallowed",
    ),
    DenylistRule(
        id="kern.ptrace",
        category=DenylistCategory.KERNEL_SURFACE,
        pattern=r"\bptrace\b",
        message="ptrace is disallowed",
    ),
    DenylistRule(
        id="kern.perf_event_open",
        category=DenylistCategory.KERNEL_SURFACE,
        pattern=r"\bperf_event_open\b",
        message="perf_event_open is disallowed",
    ),
    DenylistRule(
        id="kern.kexec",
        category=DenylistCategory.KERNEL_SURFACE,
        pattern=r"\bkexec_\w+\b",
        message="kexec_* syscalls are disallowed",
    ),
    DenylistRule(
        id="kern.setuid",
        category=DenylistCategory.KERNEL_SURFACE,
        pattern=r"\bsetuid\b",
        message="setuid syscall is disallowed",
    ),
    DenylistRule(
        id="kern.setgid",
        category=DenylistCategory.KERNEL_SURFACE,
        pattern=r"\bsetgid\b",
        message="setgid syscall is disallowed",
    ),
    DenylistRule(
        id="kern.unshare",
        category=DenylistCategory.KERNEL_SURFACE,
        pattern=r"\bunshare\b",
        message="unshare is disallowed",
    ),
    DenylistRule(
        id="kern.pivot_root",
        category=DenylistCategory.KERNEL_SURFACE,
        pattern=r"\bpivot_root\b",
        message="pivot_root is disallowed",
    ),
    DenylistRule(
        id="fs.rm_rf_root",
        category=DenylistCategory.DESTRUCTIVE_FS,
        pattern=r"\brm\s+-[rRfF]+\s+/(?:\*|\s|$)",
        message="recursive force removal of root is disallowed",
        severity="critical",
    ),
    DenylistRule(
        id="fs.mkfs",
        category=DenylistCategory.DESTRUCTIVE_FS,
        pattern=r"\bmkfs(?:\.\w+)?\b",
        message="filesystem formatting (mkfs) is disallowed",
        severity="critical",
    ),
    DenylistRule(
        id="fs.dd_zero",
        category=DenylistCategory.DESTRUCTIVE_FS,
        pattern=r"\bdd\b.*\bif=/dev/(?:zero|random|urandom)\b.*\bof=/dev/",
        message="dd to a block device is disallowed",
        severity="critical",
    ),
    DenylistRule(
        id="fs.shred",
        category=DenylistCategory.DESTRUCTIVE_FS,
        pattern=r"\bshred\b",
        message="shred is disallowed",
    ),
    DenylistRule(
        id="fs.redirect_sd",
        category=DenylistCategory.DESTRUCTIVE_FS,
        pattern=r">\s*/dev/sd[a-z]\d*",
        message="redirect to raw disk device is disallowed",
        severity="critical",
    ),
    DenylistRule(
        id="net.curl_pipe_sh",
        category=DenylistCategory.NETWORK_EXFIL,
        pattern=r"\bcurl\b[^|]*\|\s*(?:ba)?sh\b",
        message="curl piped to a shell is disallowed",
    ),
    DenylistRule(
        id="net.wget_stdout",
        category=DenylistCategory.NETWORK_EXFIL,
        pattern=r"\bwget\b[^|]*\s-O-\b",
        message="wget -O- piping is disallowed",
    ),
    DenylistRule(
        id="net.nc_listen",
        category=DenylistCategory.NETWORK_EXFIL,
        pattern=r"\bnc\b\s+.*-l\b",
        message="netcat listener is disallowed",
    ),
    DenylistRule(
        id="net.bash_revshell",
        category=DenylistCategory.NETWORK_EXFIL,
        pattern=r"bash\s+-i\s*>&\s*/dev/tcp/",
        message="bash reverse shell to /dev/tcp is disallowed",
        severity="critical",
    ),
    DenylistRule(
        id="priv.sudo_i",
        category=DenylistCategory.PRIVILEGE_ESCALATION,
        pattern=r"\bsudo\s+-[iIsS]\b",
        message="sudo interactive shell is disallowed",
    ),
    DenylistRule(
        id="priv.su_dash",
        category=DenylistCategory.PRIVILEGE_ESCALATION,
        pattern=r"(?:^|\s)su\s+-(?:\s|$)",
        message="su - is disallowed",
    ),
    DenylistRule(
        id="priv.chmod_setuid",
        category=DenylistCategory.PRIVILEGE_ESCALATION,
        pattern=r"\bchmod\s+[0-7]*4[0-7]{3}\b",
        message="setuid chmod (4xxx) is disallowed",
    ),
    DenylistRule(
        id="priv.chmod_shadow",
        category=DenylistCategory.PRIVILEGE_ESCALATION,
        pattern=r"\bchmod\b.*/etc/shadow\b",
        message="chmod on /etc/shadow is disallowed",
        severity="critical",
    ),
    DenylistRule(
        id="read.etc_shadow",
        category=DenylistCategory.SENSITIVE_READS,
        pattern=r"/etc/shadow\b",
        message="reading /etc/shadow is disallowed",
    ),
    DenylistRule(
        id="read.root_ssh",
        category=DenylistCategory.SENSITIVE_READS,
        pattern=r"/root/\.ssh/",
        message="reading /root/.ssh/ is disallowed",
    ),
    DenylistRule(
        id="read.aws_creds",
        category=DenylistCategory.SENSITIVE_READS,
        pattern=r"/(?:root|home/[^/]+)/\.aws/credentials\b",
        message="reading AWS credentials is disallowed",
    ),
    DenylistRule(
        id="read.aws_imds",
        category=DenylistCategory.SENSITIVE_READS,
        pattern=r"\b169\.254\.169\.254\b",
        message="AWS IMDS access is disallowed",
    ),
    DenylistRule(
        id="read.gcp_metadata",
        category=DenylistCategory.SENSITIVE_READS,
        pattern=r"\bmetadata\.google\.internal\b",
        message="GCP metadata server access is disallowed",
    ),
    DenylistRule(
        id="read.azure_imds",
        category=DenylistCategory.SENSITIVE_READS,
        pattern=r"Metadata:\s*true",
        message="Azure IMDS metadata header is disallowed",
    ),
    DenylistRule(
        id="codex.eval",
        category=DenylistCategory.CODE_EXEC_PRIMITIVES,
        pattern=r"(?<![\w.])eval\s*\(",
        message="eval() primitive is disallowed",
    ),
    DenylistRule(
        id="codex.exec_call",
        category=DenylistCategory.CODE_EXEC_PRIMITIVES,
        pattern=r"(?<![\w.])exec\s*\(",
        message="exec() primitive is disallowed",
    ),
    DenylistRule(
        id="codex.compile",
        category=DenylistCategory.CODE_EXEC_PRIMITIVES,
        pattern=r"(?<![\w.])compile\s*\(",
        message="compile() primitive is disallowed",
        allow_override=True,
    ),
    DenylistRule(
        id="codex.dunder_import",
        category=DenylistCategory.CODE_EXEC_PRIMITIVES,
        pattern=r"__import__\s*\(",
        message="__import__() is disallowed",
    ),
    DenylistRule(
        id="codex.pickle_loads",
        category=DenylistCategory.CODE_EXEC_PRIMITIVES,
        pattern=r"\bpickle\.loads?\s*\(",
        message="pickle.load[s] is disallowed (RCE primitive)",
    ),
    DenylistRule(
        id="codex.popen_shell_true",
        category=DenylistCategory.CODE_EXEC_PRIMITIVES,
        pattern=r"subprocess\.(?:Popen|run|call|check_output)\s*\([^)]*shell\s*=\s*True",
        message="subprocess with shell=True is disallowed",
    ),
    DenylistRule(
        id="res.forkbomb_classic",
        category=DenylistCategory.RESOURCE_EXHAUSTION,
        pattern=r":\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:",
        message="classic fork-bomb pattern detected",
        severity="critical",
    ),
    DenylistRule(
        id="res.while_true_fork",
        category=DenylistCategory.RESOURCE_EXHAUSTION,
        pattern=r"while\s+true\s*;\s*do\s+\w+\s*&\s*done",
        message="infinite fork loop detected",
    ),
    DenylistRule(
        id="res.infinite_yes",
        category=DenylistCategory.RESOURCE_EXHAUSTION,
        pattern=r"\byes\b[^|]*\|\s*(?:dd|sh|bash)\b",
        message="yes piped to a shell/dd resource-exhaustion pattern",
    ),
)


def _flatten(obj: object, out: list[str]) -> None:
    if obj is None:
        return
    if isinstance(obj, (str, bytes)):
        out.append(obj.decode("utf-8", "replace") if isinstance(obj, bytes) else obj)
        return
    if isinstance(obj, dict):
        for k, v in obj.items():
            out.append(str(k))
            _flatten(v, out)
        return
    if isinstance(obj, (list, tuple, set, frozenset)):
        for v in obj:
            _flatten(v, out)
        return
    out.append(str(obj))


class ToolSandboxDenylist:
    """Pre-dispatch policy guard over tool invocations."""

    def __init__(
        self,
        rules: list[DenylistRule] | None = None,
        strict: bool = True,
    ) -> None:
        self._rules: list[DenylistRule] = list(
            rules if rules is not None else DEFAULT_DENYLIST
        )
        self.strict = bool(strict)
        self._compiled: dict[str, re.Pattern[str]] = {}
        for rule in self._rules:
            self._compile(rule)

    def _compile(self, rule: DenylistRule) -> None:
        try:
            self._compiled[rule.id] = re.compile(rule.pattern, re.IGNORECASE)
        except re.error as exc:
            raise ValueError(
                f"invalid regex for rule {rule.id!r}: {exc}"
            ) from exc

    def add_rule(self, rule: DenylistRule) -> None:
        if any(r.id == rule.id for r in self._rules):
            raise ValueError(f"duplicate rule id: {rule.id}")
        self._compile(rule)
        self._rules.append(rule)

    def remove_rule(self, rule_id: str) -> bool:
        for i, r in enumerate(self._rules):
            if r.id == rule_id:
                del self._rules[i]
                self._compiled.pop(rule_id, None)
                return True
        return False

    @property
    def rules(self) -> tuple[DenylistRule, ...]:
        return tuple(self._rules)

    def _haystack(self, tool_name: str, tool_args: dict | str | None) -> str:
        parts: list[str] = [tool_name or ""]
        if isinstance(tool_args, str):
            parts.append(tool_args)
        elif isinstance(tool_args, dict):
            _flatten(tool_args, parts)
        elif tool_args is None:
            pass
        else:
            _flatten(tool_args, parts)
        return "\n".join(p for p in parts if p)

    def evaluate(
        self,
        tool_name: str,
        tool_args: dict | str | None,
        skip_ids: Iterable[str] | None = None,
    ) -> DenyVerdict:
        haystack = self._haystack(tool_name, tool_args)
        skip = set(skip_ids or ())
        violated: list[DenylistRule] = []
        for rule in self._rules:
            if rule.id in skip and rule.allow_override:
                continue
            pat = self._compiled[rule.id]
            if pat.search(haystack):
                violated.append(rule)
        if not violated:
            return DenyVerdict(allowed=True, violated_rules=(), advice="ok")
        advice = "; ".join(f"{r.id}: {r.message}" for r in violated)
        allowed = not self.strict
        return DenyVerdict(
            allowed=allowed,
            violated_rules=tuple(violated),
            advice=advice,
        )

    def with_overrides(
        self,
        tool_name: str,
        tool_args: dict | str | None,
        override_ids: set[str],
    ) -> DenyVerdict:
        return self.evaluate(tool_name, tool_args, skip_ids=override_ids)
