"""Aurelius CLI — Runtime diagnostics and doctor command."""

from __future__ import annotations

import importlib
import importlib.util
import os
import platform
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

logger = __import__("logging").getLogger(__name__)

# ---------------------------------------------------------------------------
# Diagnostic check types
# ---------------------------------------------------------------------------

SEVERITY_OK = "ok"
SEVERITY_WARN = "warn"
SEVERITY_ERROR = "error"
SEVERITY_INFO = "info"


@dataclass
class DiagnosticCheck:
    """Result of a single diagnostic check."""

    name: str
    severity: str
    message: str
    details: str = ""
    remediation: str = ""


@dataclass
class DoctorReport:
    """Aggregate diagnostic report."""

    host: str = ""
    platform: str = ""
    python_version: str = ""
    working_dir: str = ""
    checks: list[DiagnosticCheck] = field(default_factory=list)

    @property
    def ok_count(self) -> int:
        return sum(1 for c in self.checks if c.severity == SEVERITY_OK)

    @property
    def warn_count(self) -> int:
        return sum(1 for c in self.checks if c.severity == SEVERITY_WARN)

    @property
    def error_count(self) -> int:
        return sum(1 for c in self.checks if c.severity == SEVERITY_ERROR)

    @property
    def info_count(self) -> int:
        return sum(1 for c in self.checks if c.severity == SEVERITY_INFO)

    @property
    def healthy(self) -> bool:
        return self.error_count == 0


# ---------------------------------------------------------------------------
# Diagnostic functions
# ---------------------------------------------------------------------------

def _check_python_version() -> DiagnosticCheck:
    """Check Python version meets minimum requirements."""
    major, minor = sys.version_info[:2]
    if major >= 3 and minor >= 11:
        return DiagnosticCheck(
            name="Python version",
            severity=SEVERITY_OK,
            message=f"Python {major}.{minor} meets requirements (>= 3.11)",
        )
    if major >= 3 and minor >= 10:
        return DiagnosticCheck(
            name="Python version",
            severity=SEVERITY_WARN,
            message=f"Python {major}.{minor} works but 3.11+ recommended",
        )
    return DiagnosticCheck(
        name="Python version",
        severity=SEVERITY_ERROR,
        message=f"Python {major}.{minor} is too old (requires >= 3.10)",
        remediation="Upgrade to Python 3.11+ (3.12 recommended)",
    )


def _check_aurelius_install() -> DiagnosticCheck:
    """Check if aurelius package is importable."""
    try:
        import src  # noqa: F401
        return DiagnosticCheck(
            name="Aurelius package",
            severity=SEVERITY_OK,
            message="Aurelius src package is importable",
        )
    except ImportError:
        return DiagnosticCheck(
            name="Aurelius package",
            severity=SEVERITY_WARN,
            message="Aurelius src not on PYTHONPATH",
            remediation="Run from project root or set PYTHONPATH",
        )


def _check_core_deps() -> list[DiagnosticCheck]:
    """Check core Python dependencies."""
    checks: list[DiagnosticCheck] = []
    core_deps = {
        "yaml": "pyyaml",
        "prompt_toolkit": "prompt_toolkit",
        "rich": "rich",
        "psutil": "psutil",
    }
    for import_name, pip_name in core_deps.items():
        if importlib.util.find_spec(import_name) is not None:
            checks.append(DiagnosticCheck(
                name=f"Dependency: {pip_name}",
                severity=SEVERITY_OK,
                message=f"{pip_name} is installed",
            ))
        else:
            checks.append(DiagnosticCheck(
                name=f"Dependency: {pip_name}",
                severity=SEVERITY_WARN,
                message=f"{pip_name} is NOT installed",
                remediation=f"pip install {pip_name}",
            ))
    return checks


def _check_ml_frameworks() -> list[DiagnosticCheck]:
    """Check ML backend availability."""
    checks: list[DiagnosticCheck] = []
    frameworks = {
        "torch": {"label": "PyTorch", "desc": "ML backend"},
        "mlx": {"label": "MLX", "desc": "Apple Silicon backend"},
        "llama_cpp": {"label": "llama-cpp-python", "desc": "GGUF inference"},
        "onnxruntime": {"label": "ONNX Runtime", "desc": "ONNX inference"},
    }
    for mod_name, details in frameworks.items():
        label = details["label"]
        desc = details["desc"]
        if importlib.util.find_spec(mod_name) is not None:
            mod = importlib.import_module(mod_name)
            version = getattr(mod, "__version__", "unknown")
            checks.append(DiagnosticCheck(
                name=f"Framework: {label}",
                severity=SEVERITY_OK,
                message=f"{label} {version} available ({desc})",
            ))
        else:
            checks.append(DiagnosticCheck(
                name=f"Framework: {label}",
                severity=SEVERITY_INFO,
                message=f"{label} not installed ({desc})",
            ))
    return checks


def _check_cuda() -> DiagnosticCheck:
    """Check CUDA availability."""
    if platform.system() == "Windows":
        pass  # Check differently
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0) if device_count else "unknown"
            return DiagnosticCheck(
                name="CUDA",
                severity=SEVERITY_OK,
                message=f"CUDA available — {device_count} device(s): {device_name}",
                details=f"CUDA version: {torch.version.cuda}",
            )
        return DiagnosticCheck(
            name="CUDA",
            severity=SEVERITY_INFO,
            message="CUDA not available (no GPU or torch not built with CUDA)",
        )
    except ImportError:
        return DiagnosticCheck(
            name="CUDA",
            severity=SEVERITY_INFO,
            message="PyTorch not installed, cannot check CUDA",
        )


def _check_mps() -> DiagnosticCheck:
    """Check Metal Performance Shaders (Apple Silicon)."""
    if platform.system() != "Darwin":
        return DiagnosticCheck(
            name="MPS (Metal)",
            severity=SEVERITY_INFO,
            message="Not applicable on this platform",
        )
    try:
        import torch
        if torch.backends.mps.is_available():
            return DiagnosticCheck(
                name="MPS (Metal)",
                severity=SEVERITY_OK,
                message="MPS available — Apple Silicon GPU acceleration active",
            )
        return DiagnosticCheck(
            name="MPS (Metal)",
            severity=SEVERITY_WARN,
            message="MPS not available (torch built without MPS or restricted)",
        )
    except ImportError:
        return DiagnosticCheck(
            name="MPS (Metal)",
            severity=SEVERITY_INFO,
            message="PyTorch not installed, cannot check MPS",
        )


def _check_memory() -> list[DiagnosticCheck]:
    """Check system memory."""
    checks: list[DiagnosticCheck] = []
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024 ** 3)
        avail_gb = mem.available / (1024 ** 3)
        pct = mem.percent

        severity = SEVERITY_WARN if pct > 85 else SEVERITY_OK
        checks.append(DiagnosticCheck(
            name="System RAM",
            severity=severity,
            message=f"{avail_gb:.1f}GB available of {total_gb:.1f}GB ({pct}% used)",
        ))

        swap = psutil.swap_memory()
        if swap.total > 0:
            swap_used_pct = swap.percent
            checks.append(DiagnosticCheck(
                name="Swap",
                severity=SEVERITY_INFO,
                message=f"Swap: {swap.used / (1024**3):.1f}GB / {swap.total / (1024**3):.1f}GB ({swap_used_pct}%)",
            ))
    except ImportError:
        checks.append(DiagnosticCheck(
            name="System RAM",
            severity=SEVERITY_WARN,
            message="psutil not installed — cannot check memory",
            remediation="pip install psutil",
        ))
    return checks


def _check_disk_space() -> DiagnosticCheck:
    """Check disk space on working directory."""
    try:
        stat = os.statvfs(os.getcwd())
        free_gb = (stat.f_bavail * stat.f_frsize) / (1024 ** 3)
        severity = SEVERITY_WARN if free_gb < 5 else SEVERITY_OK
        return DiagnosticCheck(
            name="Disk space",
            severity=severity,
            message=f"{free_gb:.1f}GB free on working directory volume",
            remediation="Free disk space if below 5GB" if free_gb < 5 else "",
        )
    except OSError:
        return DiagnosticCheck(
            name="Disk space",
            severity=SEVERITY_WARN,
            message="Cannot determine disk space",
        )


def _check_artifacts_dir() -> DiagnosticCheck:
    """Check if model artifacts directory exists and has content."""
    artifacts_dirs = [
        Path("artifacts"),
        Path("models"),
        Path(".aurelius/artifacts"),
        Path.home() / ".aurelius" / "artifacts",
    ]
    for art_dir in artifacts_dirs:
        if art_dir.exists() and art_dir.is_dir():
            contents = list(art_dir.iterdir())
            if contents:
                return DiagnosticCheck(
                    name="Model artifacts",
                    severity=SEVERITY_OK,
                    message=f"Artifacts directory found with {len(contents)} item(s): {art_dir}",
                )
            return DiagnosticCheck(
                name="Model artifacts",
                severity=SEVERITY_INFO,
                message=f"Artifacts directory exists but is empty: {art_dir}",
            )
    return DiagnosticCheck(
        name="Model artifacts",
        severity=SEVERITY_WARN,
        message="No model artifacts directory found",
        remediation="Download model artifacts with `aurelius models pull`",
    )


def _check_skills_dir() -> DiagnosticCheck:
    """Check if skills directory exists."""
    skills_dir = Path("src/skills/builtin")
    if skills_dir.exists():
        skill_count = len([d for d in skills_dir.iterdir() if d.is_dir()])
        return DiagnosticCheck(
            name="Skills system",
            severity=SEVERITY_OK,
            message=f"Skills directory found with {skill_count} categories",
        )
    return DiagnosticCheck(
        name="Skills system",
        severity=SEVERITY_WARN,
        message="Skills directory not found",
    )


def _check_node() -> DiagnosticCheck:
    """Check Node.js availability for BFF layer."""
    try:
        result = subprocess.run(
            ["node", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return DiagnosticCheck(
                name="Node.js",
                severity=SEVERITY_OK,
                message=f"Node.js {result.stdout.strip()} available",
            )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return DiagnosticCheck(
        name="Node.js",
        severity=SEVERITY_INFO,
        message="Node.js not found (required for BFF layer)",
    )


def _check_rust() -> DiagnosticCheck:
    """Check Rust toolchain availability."""
    try:
        result = subprocess.run(
            ["rustc", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return DiagnosticCheck(
                name="Rust toolchain",
                severity=SEVERITY_OK,
                message=f"Rust {result.stdout.strip()}",
            )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return DiagnosticCheck(
        name="Rust toolchain",
        severity=SEVERITY_INFO,
        message="Rust not found (required for engine builds)",
    )


def _check_git() -> DiagnosticCheck:
    """Check git availability."""
    try:
        result = subprocess.run(
            ["git", "--version"], capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            return DiagnosticCheck(
                name="Git",
                severity=SEVERITY_OK,
                message=result.stdout.strip(),
            )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass
    return DiagnosticCheck(
        name="Git",
        severity=SEVERITY_ERROR,
        message="git not found",
        remediation="Install git",
    )


def _check_env_vars() -> list[DiagnosticCheck]:
    """Check relevant environment variables."""
    checks: list[DiagnosticCheck] = []
    relevant_vars = ["AURELIUS_CONFIG_DIR", "AURELIUS_DATA_DIR", "AURELIUS_LOG_DIR"]
    for var in relevant_vars:
        val = os.environ.get(var)
        if val:
            p = Path(val)
            if p.exists():
                checks.append(DiagnosticCheck(
                    name=f"Env: {var}",
                    severity=SEVERITY_OK,
                    message=f"{var}={val} (exists)",
                ))
            else:
                checks.append(DiagnosticCheck(
                    name=f"Env: {var}",
                    severity=SEVERITY_WARN,
                    message=f"{var}={val} (directory does not exist)",
                ))
    return checks


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_doctor(verbose: bool = False) -> DoctorReport:
    """Run full diagnostic suite.

    Args:
        verbose: Include INFO-level checks in output.

    Returns:
        DoctorReport with all checks and summary.
    """
    report = DoctorReport(
        host=platform.node(),
        platform=f"{platform.system()} {platform.release()} ({platform.machine()})",
        python_version=f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        working_dir=os.getcwd(),
    )

    report.checks.append(_check_python_version())
    report.checks.append(_check_aurelius_install())
    report.checks.extend(_check_core_deps())
    report.checks.extend(_check_ml_frameworks())
    report.checks.append(_check_cuda())
    report.checks.append(_check_mps())
    report.checks.extend(_check_memory())
    report.checks.append(_check_disk_space())
    report.checks.append(_check_artifacts_dir())
    report.checks.append(_check_skills_dir())
    report.checks.append(_check_node())
    report.checks.append(_check_rust())
    report.checks.append(_check_git())
    report.checks.extend(_check_env_vars())

    if not verbose:
        report.checks = [c for c in report.checks if c.severity != SEVERITY_INFO]

    return report


def format_doctor(report: DoctorReport) -> str:
    """Format a doctor report for terminal display."""
    lines: list[str] = []

    lines.append("═" * 60)
    lines.append("  Aurelius Doctor — System Diagnostics")
    lines.append("═" * 60)
    lines.append(f"  Host:            {report.host}")
    lines.append(f"  Platform:        {report.platform}")
    lines.append(f"  Python:          {report.python_version}")
    lines.append(f"  Working dir:     {report.working_dir}")
    lines.append("─" * 60)

    for check in report.checks:
        icon = {"ok": "✓", "warn": "⚠", "error": "✗", "info": "ℹ"}.get(check.severity, "?")
        lines.append(f"  {icon} {check.name}")
        lines.append(f"    {check.message}")
        if check.details:
            lines.append(f"    {check.details}")
        if check.remediation:
            lines.append(f"    → {check.remediation}")

    lines.append("─" * 60)

    health = "HEALTHY" if report.healthy else "ISSUES FOUND"
    lines.append(
        f"  Summary: {report.ok_count} ok, {report.warn_count} warnings, "
        f"{report.error_count} errors"
    )
    lines.append(f"  Status:  {health}")
    lines.append("═" * 60)

    return "\n".join(lines)
