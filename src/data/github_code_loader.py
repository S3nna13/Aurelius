"""GitHub code dataset loader for Aurelius training pipeline.

Supports CodeSearchNet, The Stack, and GitHub Issues formats.
Pure Python, no network or torch dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CodeFunction:
    """Normalized code function from CodeSearchNet."""

    repo: str
    func_name: str
    code: str
    docstring: str
    language: str
    code_tokens: list[str] = field(default_factory=list)
    url: str = ""


@dataclass
class CodeFile:
    """Normalized file from The Stack."""

    content: str
    language: str
    repo: str
    path: str
    size: int
    stars: int = 0
    avg_line_length: float = 0.0
    alphanum_fraction: float = 0.0


@dataclass
class GitHubIssue:
    number: int
    title: str
    body: str
    state: str  # "open" or "closed"
    labels: list[str]
    comments: int
    created_at: str
    user: str = ""


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_codesearchnet_sample(raw: dict) -> CodeFunction:
    """Parse a raw CodeSearchNet record into a CodeFunction."""
    return CodeFunction(
        repo=raw.get("repository_name", ""),
        func_name=raw.get("func_name", ""),
        code=raw.get("whole_func_string", raw.get("func_code_string", "")),
        docstring=raw.get("func_documentation_string", ""),
        language=raw.get("language", ""),
        code_tokens=raw.get("func_code_tokens", []),
        url=raw.get("func_code_url", ""),
    )


def parse_the_stack_sample(raw: dict) -> CodeFile:
    """Parse a raw The Stack record into a CodeFile."""
    return CodeFile(
        content=raw.get("content", ""),
        language=raw.get("lang", ""),
        repo=raw.get("max_stars_repo_name", ""),
        path=raw.get("max_stars_repo_path", ""),
        size=raw.get("size", 0),
        stars=raw.get("max_stars_count", 0),
        avg_line_length=raw.get("avg_line_length", 0.0),
        alphanum_fraction=raw.get("alphanum_fraction", 0.0),
    )


def parse_github_issue(raw: dict) -> GitHubIssue:
    """Parse a raw GitHub issue export record into a GitHubIssue."""
    raw_labels = raw.get("labels", [])
    labels: list[str] = []
    for label in raw_labels:
        if isinstance(label, dict):
            labels.append(label.get("name", ""))
        else:
            labels.append(str(label))

    user_field = raw.get("user", "")
    user = user_field.get("login", "") if isinstance(user_field, dict) else str(user_field)

    return GitHubIssue(
        number=raw.get("number", 0),
        title=raw.get("title", ""),
        body=raw.get("body", ""),
        state=raw.get("state", "open"),
        labels=labels,
        comments=raw.get("comments", 0),
        created_at=raw.get("created_at", ""),
        user=user,
    )


# ---------------------------------------------------------------------------
# Instruction converters
# ---------------------------------------------------------------------------


def code_to_instruction(fn: CodeFunction) -> dict:
    """Convert a CodeFunction to an instruction-tuning sample.

    instruction: 'Write a Python function called {func_name}'
    input:       fn.docstring
    output:      fn.code
    """
    return {
        "instruction": f"Write a Python function called {fn.func_name}",
        "input": fn.docstring,
        "output": fn.code,
    }


def issue_to_instruction(issue: GitHubIssue) -> dict:
    """Convert a GitHubIssue to an instruction-tuning sample.

    instruction: 'Analyze this GitHub issue and provide a solution'
    input:       'Title: {title}\\n{body}'
    output:      'This issue ({state}) has {comments} comments and labels: {labels}'
    """
    return {
        "instruction": "Analyze this GitHub issue and provide a solution",
        "input": f"Title: {issue.title}\n{issue.body}",
        "output": (
            f"This issue ({issue.state}) has {issue.comments} comments and labels: {issue.labels}"
        ),
    }


# ---------------------------------------------------------------------------
# Filters & deduplication
# ---------------------------------------------------------------------------


def filter_by_language(files: list[CodeFile], language: str) -> list[CodeFile]:
    """Return only files whose language matches (case-insensitive)."""
    target = language.lower()
    return [f for f in files if f.language.lower() == target]


def filter_by_quality(
    files: list[CodeFile],
    min_stars: int = 10,
    min_alphanum: float = 0.5,
) -> list[CodeFile]:
    """Keep only high-quality files (sufficient stars and alphanumeric content)."""
    return [f for f in files if f.stars >= min_stars and f.alphanum_fraction >= min_alphanum]


def deduplicate_by_content(functions: list[CodeFunction]) -> list[CodeFunction]:
    """Remove exact duplicate code strings, keeping first occurrence."""
    seen: set[str] = set()
    result: list[CodeFunction] = []
    for fn in functions:
        if fn.code not in seen:
            seen.add(fn.code)
            result.append(fn)
    return result


# ---------------------------------------------------------------------------
# Mock data generators
# ---------------------------------------------------------------------------


def mock_codesearchnet_data(n: int = 4) -> list[dict]:
    """Generate n synthetic CodeSearchNet-format records."""
    samples = []
    languages = ["python", "javascript", "java", "go"]
    for i in range(n):
        lang = languages[i % len(languages)]
        samples.append(
            {
                "repository_name": f"owner{i}/repo{i}",
                "func_path_in_repository": f"src/module{i}.py",
                "func_name": f"function_{i}",
                "whole_func_string": f"def function_{i}():\n    return {i}",
                "language": lang,
                "func_documentation_string": f"Return the value {i}.",
                "func_code_string": f"def function_{i}():\n    return {i}",
                "func_code_tokens": ["def", f"function_{i}", "(", ")", ":", "return", str(i)],
                "func_documentation_tokens": ["Return", "the", "value", str(i)],
                "split_name": "train",
                "func_code_url": f"https://github.com/owner{i}/repo{i}/blob/main/src/module{i}.py",
            }
        )
    return samples


def mock_the_stack_data(n: int = 4) -> list[dict]:
    """Generate n synthetic The Stack-format records."""
    samples = []
    languages = ["Python", "JavaScript", "Java", "Go"]
    for i in range(n):
        lang = languages[i % len(languages)]
        content = f"# File {i}\ndef hello_{i}():\n    pass\n"
        samples.append(
            {
                "content": content,
                "size": len(content),
                "lang": lang,
                "max_stars_repo_name": f"owner{i}/repo{i}",
                "max_stars_repo_path": f"src/file_{i}.py",
                "max_stars_count": (i + 1) * 25,
                "max_stars_repo_head_hexsha": f"abc{i:03d}",
                "avg_line_length": 20.0 + i,
                "max_line_length": 80,
                "alphanum_fraction": 0.6 + (i * 0.05),
            }
        )
    return samples


def mock_github_issues(n: int = 4) -> list[dict]:
    """Generate n synthetic GitHub issue export records."""
    samples = []
    states = ["open", "closed"]
    for i in range(n):
        samples.append(
            {
                "id": 10000 + i,
                "number": i + 1,
                "title": f"Issue {i}: Something went wrong",
                "body": f"Description of issue {i}. Steps to reproduce...",
                "state": states[i % len(states)],
                "labels": [{"name": "bug"}, {"name": f"priority-{i}"}],
                "comments": i * 2,
                "created_at": f"2024-01-{i + 1:02d}T00:00:00Z",
                "closed_at": f"2024-01-{i + 2:02d}T00:00:00Z" if i % 2 == 1 else None,
                "user": {"login": f"user{i}"},
            }
        )
    return samples
