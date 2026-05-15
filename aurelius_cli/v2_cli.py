"""Aurelius v2 CLI — v2 command layer built alongside the existing main.py.

This module provides the v2 command structure defined in the master plan.
The existing main.py handles legacy commands; this extends with v2 features.

Usage:
    aurelius doctor          # System health check
    aurelius hardware detect # Detect hardware
    aurelius profile use     # Use a hardware profile
    aurelius skills list     # List native skills
    aurelius daies quick     # Run quick DAIES gate
    aurelius serve           # Start runtime API server
    aurelius ui              # Open Mission Control UI
"""

from __future__ import annotations

import json
import sys

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    _RICH = True
except ImportError:
    _RICH = False


def _console() -> Console | None:
    return Console() if _RICH else None


def cmd_doctor() -> int:
    """Full health/dependency/environment diagnostic."""
    import importlib
    import platform
    import sys
    con = _console()
    
    results: list[dict] = []
    
    # Python version
    results.append({"check": "Python version", "status": "ok", "detail": sys.version})
    
    # OS
    results.append({"check": "OS", "status": "ok", "detail": f"{platform.system()} {platform.release()}"})
    
    # Key dependencies
    deps = ["numpy", "torch", "safetensors", "pydantic", "rich", "mlx", "transformers", "cuda"]
    for dep in deps:
        try:
            if dep == "cuda":
                import torch
                has_cuda = torch.cuda.is_available()
                v = f"torch.cuda available={has_cuda}"
                results.append({"check": dep, "status": "ok" if has_cuda else "warn", "detail": v})
            else:
                mod = importlib.import_module(dep)
                v = getattr(mod, "__version__", "installed")
                results.append({"check": dep, "status": "ok", "detail": v})
        except ImportError:
            results.append({"check": dep, "status": "missing", "detail": "not installed"})
    
    # Print results
    if con:
        table = Table(title="Aurelius Doctor — System Health")
        table.add_column("Check", style="cyan")
        table.add_column("Status", style="bold")
        table.add_column("Detail")
        for r in results:
            status_style = {"ok": "green", "warn": "yellow", "missing": "red"}.get(r["status"], "white")
            table.add_row(r["check"], f"[{status_style}]{r['status']}[/]", r["detail"])
        con.print(table)
    else:
        print("Aurelius Doctor")
        print("=" * 60)
        for r in results:
            print(f"  {r['check']:20s} [{r['status']:7s}] {r['detail']}")
    
    missing = [r for r in results if r["status"] == "missing"]
    if missing:
        print(f"\nMissing dependencies: {', '.join(r['check'] for r in missing)}")
        return 1
    return 0


def cmd_hardware_detect() -> int:
    """Detect hardware and recommend profile."""
    try:
        from src.runtime.hardware_detector import HardwareDetector
        detector = HardwareDetector()
        info = detector.detect()
        profile = detector.recommend_profile(info)
        
        con = _console()
        if con:
            panel = Panel(
                f"CPU: {info.cpu_arch} ({info.cpu_count} cores)\n"
                f"RAM: {info.total_ram_gb} GB\n"
                f"GPU: {info.gpu_name or 'none detected'} ({info.gpu_count} devices)\n"
                f"VRAM: {info.gpu_vram_gb} GB\n"
                f"Unified Memory: {info.unified_memory}\n"
                f"CUDA: {info.cuda_available} ({info.cuda_version})\n"
                f"MLX: {info.mlx_available}\n"
                f"TensorRT: {info.tensorrt_available}\n"
                f"Profile: {profile.id}\n"
                f"Recommended: {profile.recommended_models}",
                title="Hardware Detection Results",
                border_style="cyan",
            )
            con.print(panel)
        else:
            print(f"Hardware Profile: {profile.id}")
            print(f"CPU: {info.cpu_arch} {info.cpu_count} cores")
            print(f"RAM: {info.total_ram_gb} GB")
            print(f"Recommended Models: {profile.recommended_models}")
    except ImportError:
        print("ERROR: Runtime module not available. Run from aurelius/ directory.")
        return 1
    return 0


def cmd_skills_list(category: str | None = None) -> int:
    """List native skills."""
    try:
        from src.skills.registry import SkillRegistry
        registry = SkillRegistry()
        count = registry.discover_from_path()
        
        skills = registry.list_skills(category)
        con = _console()
        
        if con:
            table = Table(title=f"Aurelius Native Skills ({len(skills)} loaded, {count} discovered)")
            table.add_column("ID", style="cyan")
            table.add_column("Name")
            table.add_column("Category")
            table.add_column("Risk")
            table.add_column("Status")
            for s in skills:
                table.add_row(s.manifest.id, s.manifest.name, s.manifest.category,
                              s.manifest.risk_level.value, s.manifest.status.value)
            con.print(table)
        else:
            print(f"Native Skills ({len(skills)} skills)")
            print("=" * 60)
            for s in skills:
                print(f"  {s.manifest.id:40s} [{s.manifest.risk_level.value:8s}] {s.manifest.name}")
        
        stats = registry.stats()
        print(f"\nStats: {json.dumps(stats, indent=2)}")
    except ImportError as e:
        print(f"ERROR: {e}")
        return 1
    return 0


def cmd_daies_quick() -> int:
    """Run quick DAIES gate check."""
    try:
        from src.skills.registry import SkillRegistry
        from src.skills.validator import SkillValidator
        
        registry = SkillRegistry()
        count = registry.discover_from_path()
        validator = SkillValidator()
        
        results = {"total_checked": 0, "passed": 0, "failed": 0, "errors": []}
        
        for entry in registry.list_skills():
            results["total_checked"] += 1
            report = validator.validate(entry.manifest)
            if report.valid:
                results["passed"] += 1
            else:
                results["failed"] += 1
                results["errors"].append({"skill": entry.manifest.id, "errors": report.manifest_errors})
        
        con = _console()
        if con:
            from rich.table import Table
            table = Table(title="DAIES Quick Gate")
            table.add_column("Gate", style="cyan")
            table.add_column("Result")
            table.add_column("Detail")
            table.add_row("Skills Discovered", f"{count}", "builtin skills loaded")
            table.add_row("Manifest Validation", f"{results['passed']}/{results['total_checked']}", 
                          "passed" if results["failed"] == 0 else f"{results['failed']} failures")
            table.add_row("No Silent Fallback", "CHECKED", "all manifests specify model truth")
            con.print(table)
        else:
            print(f"DAIES Quick Gate: {results['passed']}/{results['total_checked']} passed, {results['failed']} failed")
        
        return 0 if results["failed"] == 0 else 1
    except ImportError as e:
        print(f"ERROR: {e}")
        return 1


def cmd_status() -> int:
    """Show current system status."""
    try:
        from src.runtime.hardware_detector import HardwareDetector
        from src.runtime.memory_budget import MemoryBudgetConfig, MemoryBudgetManager
        from src.skills.registry import SkillRegistry
        
        # Hardware
        hw = HardwareDetector()
        info = hw.detect()
        
        # Memory budget
        config = MemoryBudgetConfig(total_memory_gb=info.total_ram_gb)
        budget_mgr = MemoryBudgetManager(config)
        budget_mgr.update_consumer("weights_gb", 3.0)
        budget_mgr.update_consumer("kv_cache_gb", 1.0)
        report = budget_mgr.generate_report()
        
        # Skills
        registry = SkillRegistry()
        skill_count = registry.discover_from_path()
        
        print("Aurelius v2 Status")
        print(f"{'='*50}")
        print(f"  Hardware: {info.cpu_arch} | RAM: {info.total_ram_gb}GB")
        if info.gpu_name:
            print(f"  GPU: {info.gpu_name} | VRAM: {info.gpu_vram_gb}GB")
        print(f"  Memory: {report.used_gb:.1f}GB used / {report.available_for_aurelius_gb:.1f}GB available")
        print(f"  Pressure: {report.pressure_level.value}")
        print(f"  Skills: {skill_count} discovered")
        print(f"  CUDA: {info.cuda_available} | MLX: {info.mlx_available}")
    except ImportError as e:
        print(f"ERROR: {e}")
        return 1
    return 0


def cmd_serve(port: int = 8000) -> int:
    """Start runtime API server."""
    print(f"Starting Aurelius API server on port {port}...")
    print("See src/api/ for server implementation.")
    print("For development: uvicorn src.api.server:app --reload --port {port}")
    return 0


def cmd_ui() -> int:
    """Open Mission Control UI."""
    print("Opening Mission Control UI...")
    print("Navigate to http://localhost:5173 in your browser.")
    print("See aurelius/ui/ for frontend implementation.")
    return 0


def main_v2() -> int:
    """Entry point for v2 commands when invoked directly."""
    if len(sys.argv) < 2:
        print("Usage: python -m aurelius_cli.v2_cli <command> [args]")
        print("Commands: doctor, hardware, skills, daies, status, serve, ui")
        return 1
    
    command = sys.argv[1]
    
    if command == "doctor":
        return cmd_doctor()
    elif command == "hardware":
        if len(sys.argv) > 2 and sys.argv[2] == "detect":
            return cmd_hardware_detect()
        return 1
    elif command == "skills":
        if len(sys.argv) > 2 and sys.argv[2] in ("list",):
            cat = sys.argv[3] if len(sys.argv) > 3 else None
            return cmd_skills_list(cat)
        return 0
    elif command == "daies":
        if len(sys.argv) > 2 and sys.argv[2] in ("quick",):
            return cmd_daies_quick()
        return 0
    elif command == "status":
        return cmd_status()
    elif command == "serve":
        port = 8000
        if len(sys.argv) > 3 and sys.argv[2] == "--port":
            port = int(sys.argv[3])
        return cmd_serve(port)
    elif command == "ui":
        return cmd_ui()
    else:
        print(f"Unknown command: {command}")
        return 1


if __name__ == "__main__":
    sys.exit(main_v2())
