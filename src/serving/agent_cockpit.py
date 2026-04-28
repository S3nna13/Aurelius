"""Aurelius Agent Cockpit.

This module provides a lightweight FastAPI cockpit for agent routing,
workspace management, approval tracking, and session previews.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

try:
    from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - optional dependency fallback
    FastAPI = None
    HTTPException = Exception
    WebSocket = Any
    WebSocketDisconnect = Exception
    BaseModel = object

from src.cli.agent_router import AgentMode, AgentRouter


@dataclass(slots=True)
class WorkspaceRecord:
    id: str
    path: str
    created_at: float


@dataclass(slots=True)
class AgentPreview:
    agent_id: str
    mode: str
    name: str
    system_prompt: str


class PermissionEngine:
    def __init__(self) -> None:
        self._decisions: dict[str, dict[str, Any]] = {}

    def approve(self, action_id: str) -> None:
        self._decisions[action_id] = {
            "approved": True,
            "reason": "",
            "timestamp": time.time(),
        }

    def deny(self, action_id: str, reason: str) -> None:
        self._decisions[action_id] = {
            "approved": False,
            "reason": reason,
            "timestamp": time.time(),
        }

    def get(self, action_id: str) -> dict[str, Any] | None:
        return self._decisions.get(action_id)


class WorkspaceManager:
    def __init__(self) -> None:
        self._workspaces: dict[str, WorkspaceRecord] = {}

    def create(self, path: str) -> WorkspaceRecord:
        workspace_path = str(Path(path or os.getcwd()).resolve())
        record = WorkspaceRecord(
            id=uuid.uuid4().hex[:12],
            path=workspace_path,
            created_at=time.time(),
        )
        self._workspaces[record.id] = record
        return record

    def list(self) -> list[WorkspaceRecord]:
        return list(self._workspaces.values())


class _AgentPreviewAdapter:
    def __init__(self, agent_id: str, mode: AgentMode) -> None:
        self.agent_id = agent_id
        self.mode = mode

    async def execute(self, prompt: str) -> dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "mode": self.mode.id,
            "name": self.mode.name,
            "system_prompt": self.mode.system_prompt,
            "prompt": prompt,
        }


class CockpitRouter:
    """Route cockpit prompts to an agent mode preview."""

    def __init__(self) -> None:
        self._router = AgentRouter()

    @property
    def agents(self) -> dict[str, AgentMode]:
        return self._router.modes

    def list_agents(self) -> list[dict[str, Any]]:
        return self._router.list_modes()

    def get_adapter(self, agent_id: str) -> _AgentPreviewAdapter:
        mode = self._router.modes.get(agent_id)
        if mode is None:
            mode = self._router.classify(agent_id)
        return _AgentPreviewAdapter(agent_id=agent_id, mode=mode)


app = (
    FastAPI(
        title="Aurelius Agent Cockpit",
        version="1.0.0",
        description="Preview cockpit for agent routing, workspaces, and approvals",
    )
    if FastAPI
    else None
)

if app:
    _cors_origins_str = os.environ.get("CORS_ORIGINS", "")
    _cors_origins = (
        [origin.strip() for origin in _cors_origins_str.split(",") if origin.strip()]
        if _cors_origins_str.strip()
        else []
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )


class CreateWorkspaceRequest(BaseModel):
    path: str = ""


class SessionRequest(BaseModel):
    agent_id: str
    workspace_id: str = ""


class PromptRequest(BaseModel):
    prompt: str


class ActionRequest(BaseModel):
    action_id: str
    reason: str = ""


workspaces: dict[str, WorkspaceRecord] = {}
sessions: dict[str, dict[str, Any]] = {}
runs: list[dict[str, Any]] = []
audit_log: list[dict[str, Any]] = []

router = CockpitRouter()
permission = PermissionEngine()
workspace_mgr = WorkspaceManager()


def _workspace_payload(record: WorkspaceRecord) -> dict[str, Any]:
    return asdict(record)


if app:

    @app.post("/workspaces")
    async def create_workspace(req: CreateWorkspaceRequest):
        record = workspace_mgr.create(req.path or os.getcwd())
        workspaces[record.id] = record
        return _workspace_payload(record)

    @app.get("/workspaces")
    async def list_workspaces():
        return [_workspace_payload(record) for record in workspace_mgr.list()]

    @app.get("/workspaces/{wid}/files")
    async def list_files(wid: str):
        record = workspaces.get(wid)
        if not record:
            raise HTTPException(404, "Workspace not found")
        return {
            "files": [
                entry.name
                for entry in Path(record.path).iterdir()
                if entry.is_file()
            ][:100]
        }

    @app.get("/agents")
    async def list_agents():
        return router.list_agents()

    @app.post("/agents/{agent_id}/sessions")
    async def create_session(agent_id: str, req: SessionRequest):
        sid = uuid.uuid4().hex[:12]
        sessions[sid] = {
            "id": sid,
            "agent_id": req.agent_id or agent_id,
            "workspace_id": req.workspace_id,
            "created_at": time.time(),
            "events": [],
        }
        return sessions[sid]

    @app.post("/sessions/{sid}/prompt")
    async def send_prompt(sid: str, req: PromptRequest):
        session = sessions.get(sid)
        if not session:
            raise HTTPException(404, "Session not found")
        adapter = router.get_adapter(session["agent_id"])
        result = await adapter.execute(req.prompt)
        session["events"].append(
            {
                "type": "agent_result",
                "content": result,
                "timestamp": time.time(),
            }
        )
        return {"session_id": sid, "result": result}

    @app.post("/sessions/{sid}/actions/{action_id}/approve")
    async def approve_action(sid: str, action_id: str):
        permission.approve(action_id)
        return {"status": "approved"}

    @app.post("/sessions/{sid}/actions/{action_id}/deny")
    async def deny_action(sid: str, action_id: str, req: ActionRequest):
        permission.deny(action_id, req.reason)
        return {"status": "denied"}

    @app.get("/sessions/{sid}/events")
    async def get_events(sid: str):
        session = sessions.get(sid, {})
        return {"events": session.get("events", [])}

    @app.get("/runs")
    async def list_runs():
        return {"runs": runs[-50:]}

    @app.get("/audit")
    async def get_audit_log():
        return {"audit_log": audit_log[-100:]}

    @app.get("/health")
    async def health():
        return {"status": "ok", "agents": len(router.agents)}

    @app.websocket("/ws/sessions/{sid}/events")
    async def ws_session_events(websocket: WebSocket, sid: str):
        await websocket.accept()
        session = sessions.get(sid, {"id": sid, "events": [], "agent_id": "coding"})
        try:
            while True:
                data = await websocket.receive_text()
                msg = json.loads(data)
                if msg.get("type") == "prompt":
                    adapter = router.get_adapter(session.get("agent_id", "coding"))
                    result = await adapter.execute(msg["content"])
                    event = {
                        "type": "agent_result",
                        "content": result,
                        "timestamp": time.time(),
                    }
                    session["events"].append(event)
                    await websocket.send_json(event)
        except WebSocketDisconnect:
            pass


def start_server(host: str = "127.0.0.1", port: int = 8080) -> None:
    """Start the Agent Cockpit API server."""
    import uvicorn

    uvicorn.run(app, host=host, port=port)

