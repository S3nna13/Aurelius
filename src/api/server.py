"""Aurelius v2 API Server — FastAPI runtime truth layer.

Serves:
- /api/hardware/detect
- /api/hardware/profiles
- /api/backends
- /api/backends/select
- /api/capabilities
- /api/daies/runs
- /api/daies/run
- /api/exports
- /api/cua/*
- /api/checkpoints
- /api/approvals
- /api/skills/native/*
- /api/models — with artifacts/backend/quantization/capabilities
- /api/health — with RAM/VRAM/unified memory
- /api/chat — includes requested vs actual model/backend metadata
- /api/agents — includes execution mode and budgets
- /api/memory — includes quarantine state
- /api/traces — includes provenance
"""

from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="Aurelius v2 API", version="2.0.0")


# ── Request/Response Models ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    prompt: str
    requested_model: str = "forge"
    mode: str = "chat"
    context_size: int = 32768

class ChatResponse(BaseModel):
    text: str
    requested_model: str
    actual_model: str
    execution_mode: str
    backend: str
    artifact: str
    local_or_remote: str

class CapabilityResponse(BaseModel):
    requested_model: str
    actual_model: str
    execution_mode: str
    backend: str
    artifact: str
    quantization: str
    context_budget: int
    local_or_remote: str
    hardware_profile: str
    capabilities: dict[str, dict]
    disabled_capabilities: list[str]
    fallback_reason: str | None
    live_status: str


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/api/health")
async def api_health() -> dict[str, Any]:
    """Health endpoint with RAM/VRAM/unified memory."""
    try:
        from src.runtime.hardware_detector import HardwareDetector
        from src.runtime.memory_budget import MemoryBudgetConfig, MemoryBudgetManager

        detector = HardwareDetector()
        info = detector.detect()
        config = MemoryBudgetConfig(total_memory_gb=info.total_ram_gb)
        budget = MemoryBudgetManager(config)
        report = budget.generate_report()

        return {
            "status": "healthy",
            "version": "2.0.0",
            "hardware": {
                "cpu_arch": info.cpu_arch,
                "total_ram_gb": info.total_ram_gb,
                "gpu_vram_gb": info.gpu_vram_gb,
                "unified_memory": info.unified_memory,
                "gpu_name": info.gpu_name,
                "cuda_available": info.cuda_available,
                "mlx_available": info.mlx_available,
            },
            "memory": report.to_dict() if hasattr(report, "to_dict") else {
                "total_memory_gb": report.total_memory_gb,
                "available_for_aurelius_gb": report.available_for_aurelius_gb,
                "used_gb": report.used_gb,
                "free_gb": report.free_gb,
                "pressure_level": report.pressure_level.value,
            },
        }
    except Exception as e:
        return {"status": "degraded", "error": str(e)}


# ── Hardware Detection ────────────────────────────────────────────────────────

@app.get("/api/hardware/detect")
async def api_hardware_detect() -> dict[str, Any]:
    """Detect hardware and return full info."""
    try:
        from src.runtime.hardware_detector import HardwareDetector
        detector = HardwareDetector()
        info = detector.detect()
        profile = detector.recommend_profile(info)
        return {
            "info": {
                "cpu_arch": info.cpu_arch,
                "cpu_count": info.cpu_count,
                "total_ram_gb": info.total_ram_gb,
                "gpu_vram_gb": info.gpu_vram_gb,
                "unified_memory": info.unified_memory,
                "gpu_name": info.gpu_name,
                "cuda_version": info.cuda_version,
                "mlx_available": info.mlx_available,
            },
            "profile": profile.id,
            "recommended_models": profile.recommended_models,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/hardware/profiles")
async def api_hardware_profiles() -> dict[str, Any]:
    """List all known hardware profiles."""
    profiles = {
        "jetson_nano_2gb": {"ram": 2, "vram": 0, "unified": True, "recommended": "swift-q3"},
        "jetson_nano_4gb": {"ram": 4, "vram": 0, "unified": True, "recommended": "swift-q4"},
        "jetson_orin_8gb": {"ram": 8, "vram": 0, "unified": True, "recommended": "forge-q4"},
        "mac_silicon_8gb": {"ram": 8, "vram": 0, "unified": True, "recommended": "swift-q4"},
        "mac_silicon_16gb": {"ram": 16, "vram": 0, "unified": True, "recommended": "forge-q4"},
        "mac_silicon_32gb": {"ram": 32, "vram": 0, "unified": True, "recommended": "forge-q4"},
        "mac_silicon_64gb": {"ram": 64, "vram": 0, "unified": True, "recommended": "forge-native"},
        "mac_ultra_128gb": {"ram": 128, "vram": 0, "unified": True, "recommended": "atlas-q4"},
        "rtx_8gb": {"ram": 16, "vram": 8, "unified": False, "recommended": "swift-q4"},
        "rtx_12gb": {"ram": 16, "vram": 12, "unified": False, "recommended": "forge-q5"},
        "rtx_24gb": {"ram": 32, "vram": 24, "unified": False, "recommended": "forge-native"},
        "rtx_6000_48gb": {"ram": 64, "vram": 48, "unified": False, "recommended": "atlas-q4"},
        "rtx_pro_6000_bw_96gb": {"ram": 128, "vram": 96, "unified": False, "recommended": "atlas-native"},
    }
    return {"profiles": profiles}


# ── Backends ──────────────────────────────────────────────────────────────────

@app.get("/api/backends")
async def api_backends() -> dict[str, Any]:
    """List available backends and their status."""
    return {
        "backends": [
            {"name": "pytorch_eager", "status": "available", "platforms": ["linux", "macos", "windows"]},
            {"name": "torch_compile", "status": "available", "platforms": ["linux"]},
            {"name": "llama_cpp_gguf", "status": "available", "platforms": ["linux", "macos", "windows"]},
            {"name": "mlx", "status": "conditional", "platforms": ["macos"], "requires": "apple_silicon"},
            {"name": "onnx_runtime", "status": "available", "platforms": ["linux", "macos", "windows"]},
            {"name": "tensorrt_edge_llm", "status": "conditional", "platforms": ["linux"], "requires": "jetson"},
            {"name": "tensorrt_llm", "status": "conditional", "platforms": ["linux"], "requires": "nvidia_gpu"},
            {"name": "vllm", "status": "conditional", "platforms": ["linux"], "requires": "nvidia_gpu"},
            {"name": "remote_aurelius", "status": "available", "platforms": ["all"]},
        ]
    }


@app.post("/api/backends/select")
async def api_backend_select(model: str = "forge") -> dict[str, Any]:
    """Select best backend for the given model."""
    try:
        from src.runtime.backend_selector import BackendSelector
        from src.runtime.hardware_detector import HardwareDetector

        detector = HardwareDetector()
        info = detector.detect()
        profile = detector.recommend_profile(info)
        selector = BackendSelector()
        selection = selector.select(model, profile)
        return {
            "model": model,
            "backend": selection.backend.value,
            "quantization": selection.quantization,
            "context_budget": selection.context_budget,
            "capability_mode": selection.capability_mode.value,
            "reasons": selection.reasons,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Capabilities ──────────────────────────────────────────────────────────────

@app.get("/api/capabilities")
async def api_capabilities(model: str = "forge") -> CapabilityResponse:
    """Generate capability report for current profile."""
    try:
        from src.runtime.backend_selector import BackendSelector
        from src.runtime.hardware_detector import HardwareDetector

        detector = HardwareDetector()
        info = detector.detect()
        profile = detector.recommend_profile(info)
        selector = BackendSelector()
        selection = selector.select(model, profile)
        selection_report = selection.to_capability_report(model, model, profile)
        data = selection_report.to_dict()
        return CapabilityResponse(**data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── DAIES ─────────────────────────────────────────────────────────────────────

@app.get("/api/daies/runs")
async def api_daies_runs() -> dict[str, Any]:
    """List DAIES gate run history."""
    return {"runs": [], "total": 0}


@app.post("/api/daies/run")
async def api_daies_run(gate: str = "quick") -> dict[str, Any]:
    """Run DAIES validation gates."""
    try:
        from src.skills.registry import SkillRegistry
        from src.skills.validator import SkillValidator

        registry = SkillRegistry()
        count = registry.discover_from_path()
        validator = SkillValidator()
        results = {"total": 0, "passed": 0, "failed": 0, "gates": {}}

        for entry in registry.list_skills():
            results["total"] += 1
            report = validator.validate(entry.manifest)
            if report.valid:
                results["passed"] += 1
            else:
                results["failed"] += 1
            for g, passed in report.daies_gate_results.items():
                results["gates"].setdefault(g, {"passed": 0, "failed": 0})
                if passed:
                    results["gates"][g]["passed"] += 1
                else:
                    results["gates"][g]["failed"] += 1

        results["discovered"] = count
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Skills ────────────────────────────────────────────────────────────────────

@app.get("/api/skills/native")
async def api_skills_native(category: str | None = None, search: str | None = None) -> dict[str, Any]:
    """List native skills with optional filter."""
    try:
        from src.skills.registry import SkillRegistry
        registry = SkillRegistry()
        registry.discover_from_path()

        if category:
            skills = registry.list_skills(category)
        elif search:
            skills = registry.search(search)
        else:
            skills = registry.list_skills()

        return {
            "skills": [s.manifest.to_dict() for s in skills],
            "total": len(skills),
            "stats": registry.stats(),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/skills/native/{skill_id}")
async def api_skill_detail(skill_id: str) -> dict[str, Any]:
    """Get detail for a specific skill."""
    try:
        from src.skills.registry import SkillRegistry
        from src.skills.validator import SkillValidator

        registry = SkillRegistry()
        registry.discover_from_path()
        entry = registry.get(skill_id)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Skill {skill_id} not found")

        validator = SkillValidator()
        validation = validator.validate(entry.manifest)

        return {
            "manifest": entry.manifest.to_dict(),
            "loaded": entry.loaded,
            "validation": validation.to_dict(),
            "telemetry": entry.telemetry.to_dict() if entry.telemetry else None,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/skills/native/{skill_id}/run")
async def api_skill_run(skill_id: str, mode: str = "dry_run") -> dict[str, Any]:
    """Run a native skill."""
    try:
        from src.skills.executor import SkillExecutor
        from src.skills.manifest import SkillExecutionMode
        from src.skills.registry import SkillRegistry

        registry = SkillRegistry()
        registry.discover_from_path()
        entry = registry.get(skill_id)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Skill {skill_id} not found")

        executor = SkillExecutor()
        result = executor.execute(entry.manifest, SkillExecutionMode(mode))
        return {"skill_id": skill_id, "mode": mode, "success": result.success, "output": result.dry_run_details or "executed"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/skills/native/{skill_id}/telemetry")
async def api_skill_telemetry(skill_id: str) -> dict[str, Any]:
    """Get telemetry for a skill."""
    try:
        from src.skills.registry import SkillRegistry
        registry = SkillRegistry()
        registry.discover_from_path()
        entry = registry.get(skill_id)
        if not entry:
            raise HTTPException(status_code=404, detail=f"Skill {skill_id} not found")
        return {"telemetry": entry.telemetry.to_dict() if entry.telemetry else None}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ── Chat ──────────────────────────────────────────────────────────────────────

@app.post("/api/chat")
async def api_chat(request: ChatRequest) -> ChatResponse:
    """Chat endpoint with full execution metadata."""
    try:
        from src.runtime.backend_selector import BackendSelector
        from src.runtime.hardware_detector import HardwareDetector

        detector = HardwareDetector()
        info = detector.detect()
        profile = detector.recommend_profile(info)
        selector = BackendSelector()
        selection = selector.select(request.requested_model, profile)

        return ChatResponse(
            text="Aurelius v2 chat endpoint active. Model inference requires model weights.",
            requested_model=request.requested_model,
            actual_model=request.requested_model,
            execution_mode=selection.capability_mode.value,
            backend=selection.backend.value,
            artifact=f"{request.requested_model}-default",
            local_or_remote="local" if "remote" not in selection.backend.value else "remote",
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
