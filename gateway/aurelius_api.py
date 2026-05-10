"""Aurelius API Server — FastAPI with WebSocket streaming.

Serves the Agent Cockpit frontend with real-time communication.
"""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

app = FastAPI(
    title="Aurelius API", version="1.0.0", description="Backend API for the Aurelius Agent Cockpit"
)

# CORS origins must be configured via CORS_ORIGINS env var (comma-separated).
# Fail closed: default to empty list (no cross-origin requests) in production.
_cors_origins_str = os.environ.get("CORS_ORIGINS", "")
_cors_origins = (
    [o.strip() for o in _cors_origins_str.split(",") if o.strip()]
    if _cors_origins_str.strip()
    else []
)
app.add_middleware(
    CORSMiddleware, allow_origins=_cors_origins, allow_methods=["*"], allow_headers=["*"]
)

# ─── Data Models ──────────────────────────────────────────


class PromptRequest(BaseModel):
    prompt: str
    session_id: str = ""
    mode: str = "coding"


class ActionRequest(BaseModel):
    action_id: str
    approved: bool
    reason: str = ""


class WorkspaceRequest(BaseModel):
    path: str = ""


# ─── State ────────────────────────────────────────────────

sessions: dict[str, dict[str, Any]] = {}
audit_log: list[dict[str, Any]] = []
workspaces: dict[str, str] = {"default": os.getcwd()}

# ─── REST Endpoints ───────────────────────────────────────


@app.get("/")
async def root():
    return {"service": "Aurelius API", "version": "1.0.0", "status": "ready"}


@app.get("/health")
async def health():
    return {"status": "ok", "uptime": time.time(), "sessions": len(sessions)}


@app.post("/workspaces")
async def create_workspace(req: WorkspaceRequest):
    wid = uuid.uuid4().hex[:8]
    workspaces[wid] = req.path or os.getcwd()
    return {"id": wid, "path": workspaces[wid]}


@app.get("/workspaces")
async def list_workspaces():
    return [{"id": k, "path": v} for k, v in workspaces.items()]


@app.post("/sessions")
async def create_session():
    sid = uuid.uuid4().hex[:12]
    sessions[sid] = {"id": sid, "events": [], "created_at": time.time()}
    return {"session_id": sid}


@app.post("/sessions/{sid}/prompt")
async def send_prompt(sid: str, req: PromptRequest):
    session = sessions.get(sid)
    if not session:
        raise HTTPException(404, "Session not found")
    event = {
        "type": "response",
        "content": f"Received: {req.prompt[:80]}...",
        "timestamp": time.time(),
    }
    session["events"].append(event)
    return {"event": event}


@app.post("/actions/approve")
async def approve_action(req: ActionRequest):
    audit_log.append(
        {
            "action_id": req.action_id,
            "approved": req.approved,
            "reason": req.reason,
            "timestamp": time.time(),
        }
    )
    return {"status": "approved" if req.approved else "denied"}


@app.get("/audit")
async def get_audit():
    return {"audit_log": audit_log[-100:]}


@app.get("/frontend")
async def get_frontend():
    fp = Path(__file__).parent.parent.parent / "frontend" / "agent_cockpit.html"
    if fp.exists():
        return FileResponse(str(fp))
    return {"error": "Frontend not found"}


# ─── WebSocket ────────────────────────────────────────────

connected_clients: dict[str, WebSocket] = {}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    client_id = uuid.uuid4().hex[:8]
    connected_clients[client_id] = ws
    try:
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)
            if msg.get("type") == "prompt":
                response = {
                    "type": "response",
                    "content": f"Echo: {msg['content'][:60]}",
                    "timestamp": time.time(),
                }
                await ws.send_json(response)
    except WebSocketDisconnect:
        connected_clients.pop(client_id, None)


def start(host: str = "127.0.0.1", port: int = 8080):
    import uvicorn

    uvicorn.run(app, host=host, port=port)
