from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class LintIssue:
    line: int
    col: int
    code: str
    message: str


@dataclass(frozen=True)
class LintResult:
    issues: tuple[LintIssue, ...]
    passed: bool


class LinterTool:
    def __init__(self) -> None:
        pass

    def lint(self, source: str) -> LintResult:
        issues: list[LintIssue] = []
        try:
            tree = ast.parse(source)
        except SyntaxError as e:
            issues.append(LintIssue(
                line=e.lineno or 0,
                col=e.offset or 0,
                code="E001",
                message=str(e.msg),
            ))
            return LintResult(issues=tuple(issues), passed=False)

        for node in ast.walk(tree):
            if isinstance(node, ast.ExceptHandler) and node.type is None:
                issues.append(LintIssue(
                    line=node.lineno,
                    col=node.col_offset,
                    code="W001",
                    message="bare except clause",
                ))
            elif isinstance(node, ast.Assert):
                issues.append(LintIssue(
                    line=node.lineno,
                    col=node.col_offset,
                    code="W002",
                    message="assert statement (use explicit check)",
                ))
            elif (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "print"
            ):
                issues.append(LintIssue(
                    line=node.lineno,
                    col=node.col_offset,
                    code="W003",
                    message="print call (use logger)",
                ))

        return LintResult(issues=tuple(issues), passed=len(issues) == 0)

    def lint_file(self, path: str | Path) -> LintResult:
        return self.lint(Path(path).read_text())


LINTER_REGISTRY: dict[str, type[LinterTool]] = {"default": LinterTool}
