# Copyright (c) 2025 Aurelius Systems, Inc. All rights reserved.
# This file is proprietary and confidential. Unauthorized use is strictly prohibited.
# See LICENSE for full terms.

"""Aurelius unified frontend server.

Serves the built React SPA from ``frontend/dist/`` and exposes a
unified REST + SSE API surface for the Aurelius agent system.

Run: python -m src.serving.aurelius_server --port 7870

Endpoints
---------
Static:
  GET  /                       → serve built React app
  GET  /assets/*               → serve Vite-generated assets
  GET  /vite.svg               → serve static assets

API (JSON):
  GET  /api/health             → health check
  GET  /api/status             → agents, skills, plugins, memory stats
  POST /api/command            → NL command dispatch
  GET  /api/activity           → activity feed
  GET  /api/notifications      → Hermes notification inbox
  POST /api/notifications/read → mark notification read
  POST /api/notifications/read-all → mark all read
  GET  /api/notifications/stats → unread counts, by-channel breakdown
  GET  /api/skills             → list skills
  GET  /api/plugins            → list plugins
  GET  /api/workflows          → list workflow definitions
  GET  /api/memory             → memory layer summary
  GET  /api/modes              → agent mode registry

Realtime:
  GET  /api/events             → SSE stream of system events
"""

from __future__ import annotations

import argparse
import json
import logging
import mimetypes
import time
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from src.agent.command_dispatcher import CommandDispatcher
from src.agent.nl_command_parser import NLCommandParseError, NLCommandParser
from src.agent.skill_executor import ExecutionResult, SkillContext, SkillExecutor
from src.serving.mission_control import ActivityLog
from src.workflow.workflow_monitor import WorkflowMonitor, WorkflowStatus


class _LogRingBuffer(logging.Handler):
    """In-memory ring buffer for recent log records."""

    def __init__(self, capacity: int = 500) -> None:
        super().__init__()
        self.capacity = capacity
        self.records: list[dict[str, Any]] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append({
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": self.format(record),
        })
        if len(self.records) > self.capacity:
            self.records = self.records[-self.capacity :]


logger = logging.getLogger(__name__)

_log_buffer = _LogRingBuffer(capacity=500)
_log_buffer.setFormatter(logging.Formatter("%(message)s"))
logger.addHandler(_log_buffer)

#: Maximum request body size (1 MiB).
_MAX_CONTENT_LENGTH = 1_048_576

#: Path to built frontend assets — resolved relative to this file.
_FRONTEND_DIST = Path(__file__).resolve().parent.parent.parent / "frontend" / "dist"


class _JSONMixin:
    """Helper for JSON request/response handling."""

    def _send_json(self, status: int, data: dict[str, Any]) -> None:
        body = json.dumps(data, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> bytes:
        try:
            content_length = int(self.headers.get("Content-Length", 0))
        except ValueError as exc:
            raise ValueError("Invalid Content-Length header") from exc
        if content_length < 0:
            raise ValueError("Negative Content-Length")
        if content_length > _MAX_CONTENT_LENGTH:
            raise ValueError(
                f"Content-Length {content_length} exceeds maximum {_MAX_CONTENT_LENGTH}"
            )
        return self.rfile.read(content_length)

    def _parse_json_body(self) -> dict[str, Any]:
        raw = self._read_body()
        if not raw:
            return {}
        return json.loads(raw)


class AureliusHandler(BaseHTTPRequestHandler, _JSONMixin):
    def log_message(self, fmt: str, *args: Any) -> None:
        logger.debug(fmt, *args)

    # ------------------------------------------------------------------
    # Static file serving
    # ------------------------------------------------------------------

    def _serve_static(self, relative_path: str) -> None:
        """Serve a file from the built frontend dist directory."""
        # Security: prevent directory traversal
        safe_path = relative_path.lstrip("/")
        if ".." in safe_path:
            self.send_response(403)
            self.end_headers()
            return

        # Security: block source code and sensitive file access
        blocked_exts = {".py", ".ts", ".tsx", ".jsx", ".map", ".env", ".git", ".lock", ".toml", ".cfg", ".ini"}
        if any(safe_path.lower().endswith(ext) for ext in blocked_exts):
            self.send_response(403)
            self.end_headers()
            return

        file_path = _FRONTEND_DIST / safe_path
        if not file_path.exists() or file_path.is_dir():
            # Fallback to index.html for SPA routing
            index_path = _FRONTEND_DIST / "index.html"
            if index_path.exists():
                file_path = index_path
            else:
                self.send_response(404)
                self.end_headers()
                return

        content_type, _ = mimetypes.guess_type(str(file_path))
        if content_type is None:
            content_type = "application/octet-stream"

        try:
            data = file_path.read_bytes()
        except OSError:
            self.send_response(500)
            self.end_headers()
            return

        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        # Cache assets with hash in filename (Vite pattern)
        if "assets/" in safe_path:
            self.send_header("Cache-Control", "public, max-age=31536000, immutable")
        self.end_headers()
        self.wfile.write(data)

    def _check_auth(self) -> bool:
        server = self.server
        if not getattr(server, "runtime_config", {}).get("require_auth", False):
            return True
        if self.path == "/api/health" or self.path == "/api/license/validate":
            return True
        api_key = self.headers.get("X-API-Key", "")
        expected = getattr(server, "runtime_config", {}).get("api_key", "")
        if expected and api_key == expected:
            return True
        session = self.headers.get("X-Session-Token", "")
        valid_sessions = getattr(server, "_valid_sessions", set())
        if session in valid_sessions:
            return True
        return False

    def _require_auth(self) -> bool:
        if not self._check_auth():
            self._send_json(401, {"error": "Unauthorized", "message": "Valid API key or session required"})
            return False
        return True

    def do_GET(self):
        # API routes
        if self.path == "/api/health":
            self._handle_health()
            return
        if self.path == "/api/status":
            self._handle_status()
            return
        if self.path.startswith("/api/agents/") and self.path.count("/") == 3:
            self._handle_agent_detail()
            return
        if self.path == "/api/activity":
            self._handle_activity()
            return
        if self.path == "/api/notifications":
            self._handle_notifications()
            return
        if self.path == "/api/notifications/stats":
            self._handle_notification_stats()
            return
        if self.path == "/api/skills":
            self._handle_skills()
            return
        if self.path.startswith("/api/skills/"):
            self._handle_skill_detail()
            return
        if self.path == "/api/plugins":
            self._handle_plugins()
            return
        if self.path == "/api/workflows":
            self._handle_workflows()
            return
        if self.path.startswith("/api/workflows/") and self.path.count("/") == 3:
            self._handle_workflow_detail()
            return
        if self.path == "/api/memory":
            self._handle_memory()
            return
        if self.path == "/api/memory/entries":
            self._handle_memory_entries()
            return
        if self.path == "/api/modes":
            self._handle_modes()
            return
        if self.path == "/api/config":
            self._handle_config_get()
            return
        if self.path == "/api/events":
            if not self._require_auth():
                return
            self._handle_sse()
            return
        if self.path == "/api/logs":
            if not self._require_auth():
                return
            self._handle_logs()
            return
        if self.path == "/api/license/validate":
            self._handle_license_validate()
            return

        if self.path.startswith("/api/") and not self._require_auth():
            return

        # Static files
        path = self.path
        if path == "/":
            path = "/index.html"
        self._serve_static(path)

    def do_POST(self):
        if self.path == "/api/license/activate":
            self._handle_license_activate()
            return
        if self.path.startswith("/api/") and not self._require_auth():
            return
        if self.path == "/api/command":
            self._handle_command()
            return
        if self.path == "/api/notifications/read":
            self._handle_notification_read()
            return
        if self.path == "/api/notifications/read-all":
            self._handle_notification_read_all()
            return
        if self.path == "/api/notifications/preferences":
            self._handle_notification_preferences_post()
            return
        if self.path == "/api/skills/execute":
            self._handle_skill_execute()
            return
        if self.path.startswith("/api/workflows/") and self.path.endswith("/trigger"):
            self._handle_workflow_trigger()
            return
        if self.path == "/api/config":
            self._handle_config_post()
            return
        self.send_response(404)
        self.end_headers()

    # ------------------------------------------------------------------
    # API handlers
    # ------------------------------------------------------------------

    def _handle_health(self) -> None:
        self._send_json(200, {"status": "ok", "time": time.time()})

    def _handle_status(self) -> None:
        server = self.server
        status: dict[str, Any] = {"agents": [], "skills": [], "plugins": []}

        supervisor = getattr(server, "supervisor", None)
        if supervisor is not None and hasattr(supervisor, "_agents"):
            for aid, rec in supervisor._agents.items():
                status["agents"].append({"id": aid, "state": str(getattr(rec, "state", "UNKNOWN"))})
        else:
            status["agents"].append({"id": "default", "state": "IDLE"})

        skill_catalog = getattr(server, "skill_catalog", None)
        if skill_catalog is not None and hasattr(skill_catalog, "list"):
            try:
                for entry in skill_catalog.list():
                    status["skills"].append(
                        {
                            "id": getattr(entry, "skill_id", str(entry)),
                            "active": getattr(entry, "active", False),
                        }
                    )
            except Exception:
                logger.warning("Status: failed to enumerate skills", exc_info=True)
        else:
            status["skills"].append({"id": "code-review", "active": True})

        plugin_loader = getattr(server, "plugin_loader", None)
        if plugin_loader is not None and hasattr(plugin_loader, "list_loaded"):
            try:
                for pid in plugin_loader.list_loaded():
                    status["plugins"].append({"id": pid})
            except Exception:
                logger.warning("Status: failed to enumerate plugins", exc_info=True)
        else:
            status["plugins"].append({"id": "mcp-core"})

        # Memory stats
        memory = getattr(server, "memory", None)
        if memory is not None and hasattr(memory, "stats"):
            try:
                status["memory"] = memory.stats()
            except Exception:  # noqa: S110
                pass
        else:
            status["memory"] = {"total_entries": 0}

        # Hermes stats
        hermes = getattr(server, "hermes", None)
        if hermes is not None:
            status["notifications"] = {"unread": hermes.unread_count()}
        else:
            status["notifications"] = {"unread": 0}

        # Aggregated counts for dashboard
        status["counts"] = {
            "agents_online": sum(
                1
                for a in status["agents"]
                if a.get("state", "").upper() in ("ACTIVE", "RUNNING", "IDLE")
            ),
            "agents_total": len(status["agents"]),
            "skills_active": sum(1 for s in status["skills"] if s.get("active", False)),
            "skills_total": len(status["skills"]),
            "plugins_total": len(status["plugins"]),
            "notifications_unread": status.get("notifications", {}).get("unread", 0),
        }
        self._send_json(200, status)

    def _handle_agent_detail(self) -> None:
        parts = self.path.split("/")
        agent_id = parts[-1] if len(parts) >= 4 else ""
        supervisor = getattr(self.server, "supervisor", None)
        if supervisor is not None and hasattr(supervisor, "_agents"):
            rec = supervisor._agents.get(agent_id)
            if rec is not None:
                self._send_json(
                    200,
                    {
                        "id": agent_id,
                        "state": str(getattr(rec, "state", "UNKNOWN")),
                        "metrics": getattr(rec, "metrics", {}),
                    },
                )
                return
        self._send_json(404, {"error": "Agent not found"})

    def _handle_activity(self) -> None:
        log = getattr(self.server, "activity_log", None)
        if log is None:
            self._send_json(200, {"entries": []})
            return
        self._send_json(200, {"entries": log.to_dicts()})

    def _handle_command(self) -> None:
        try:
            payload = self._parse_json_body()
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})
            return

        text = payload.get("command", "")
        if not isinstance(text, str):
            self._send_json(400, {"error": "command must be a string"})
            return

        server = self.server
        parser = getattr(server, "nl_parser", None) or NLCommandParser()
        dispatcher = getattr(server, "command_dispatcher", None) or CommandDispatcher()
        log = getattr(server, "activity_log", None)
        hermes = getattr(server, "hermes", None)

        try:
            parsed = parser.parse(text)
            result = dispatcher.dispatch(parsed)
        except NLCommandParseError as exc:
            if log is not None:
                log.append(text, False, str(exc))
            self._send_json(400, {"error": str(exc), "success": False})
            return
        except Exception as exc:
            logger.exception("Command dispatch error")
            if log is not None:
                log.append(text, False, str(exc))
            if hermes is not None:
                hermes.notify(
                    "Command failed",
                    str(exc),
                    category="alert",
                    priority="high",
                )
            self._send_json(500, {"error": str(exc), "success": False})
            return

        if log is not None:
            log.append(text, result.success, result.output)

        usage = getattr(server, "usage_pipeline", None)
        if usage is not None:
            try:
                usage.log_chat(
                    user_id="anonymous",
                    model="aurelius-v1",
                    tokens_in=len(text.split()),
                    tokens_out=len(result.output.split()),
                )
            except Exception:
                logger.exception("Usage logging failed")

        self._send_json(
            200,
            {
                "success": result.success,
                "output": result.output,
                "action": parsed.action,
                "target": parsed.target,
            },
        )

    def _handle_notifications(self) -> None:
        hermes = getattr(self.server, "hermes", None)
        if hermes is None:
            self._send_json(200, {"notifications": []})
            return

        query = self.path.split("?")[1] if "?" in self.path else ""
        params: dict[str, str] = {}
        if query:
            for part in query.split("&"):
                if "=" in part:
                    k, v = part.split("=", 1)
                    params[k] = v

        category = params.get("category") or None
        priority = params.get("priority") or None
        read_str = params.get("read")
        read = {"true": True, "false": False}.get(read_str.lower()) if read_str else None
        limit = int(params.get("limit", "100"))

        notifications = hermes.list_notifications(
            category=category,
            priority=priority,
            read=read,
            limit=limit,
        )

        self._send_json(
            200,
            {
                "notifications": [
                    {
                        "id": n.notification_id,
                        "timestamp": n.timestamp,
                        "channel": n.channel,
                        "priority": n.priority,
                        "category": n.category,
                        "title": n.title,
                        "body": n.body,
                        "read": n.read,
                        "delivered": n.delivered,
                    }
                    for n in notifications
                ]
            },
        )

    def _handle_notification_stats(self) -> None:
        hermes = getattr(self.server, "hermes", None)
        if hermes is None:
            self._send_json(200, {"unread": 0, "total": 0})
            return
        stats = hermes.stats()
        self._send_json(
            200,
            {
                "unread": stats.get("unread", 0),
                "total": stats.get("total", 0),
                "by_channel": stats.get("by_channel", {}),
                "by_priority": stats.get("by_priority", {}),
            },
        )

    def _handle_notification_read(self) -> None:
        hermes = getattr(self.server, "hermes", None)
        if hermes is None:
            self._send_json(200, {"success": False, "error": "Hermes not available"})
            return
        try:
            payload = self._parse_json_body()
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})
            return

        nid = payload.get("id", "")
        success = hermes.mark_read(nid)
        self._send_json(200, {"success": success})

    def _handle_notification_read_all(self) -> None:
        hermes = getattr(self.server, "hermes", None)
        if hermes is None:
            self._send_json(200, {"success": False, "error": "Hermes not available"})
            return
        try:
            payload = self._parse_json_body()
        except (ValueError, json.JSONDecodeError):
            payload = {}

        category = payload.get("category") or None
        count = hermes.mark_all_read(category=category)
        self._send_json(200, {"success": True, "count": count})

    def _handle_skills(self) -> None:
        catalog = getattr(self.server, "skill_catalog", None)
        skills: list[dict[str, Any]] = []
        if catalog is not None and hasattr(catalog, "list"):
            try:
                for entry in catalog.list():
                    skills.append(
                        {
                            "id": getattr(entry, "skill_id", str(entry)),
                            "name": getattr(entry, "name", "unknown"),
                            "description": getattr(entry, "description", ""),
                            "scope": getattr(entry, "scope", "repo"),
                            "active": getattr(entry, "active", False),
                            "version": getattr(entry, "version", None),
                            "risk_score": getattr(entry, "risk_score", 0.0),
                            "allow_level": getattr(entry, "allow_level", "review"),
                            "category": getattr(entry, "source_kind", "local"),
                        }
                    )
            except Exception:
                logger.warning("Skills: failed to enumerate", exc_info=True)
        else:
            skills = [
                {
                    "id": "code-review",
                    "name": "Code Review",
                    "description": "Automated code review and suggestions.",
                    "scope": "repo",
                    "active": True,
                    "version": "1.0.0",
                    "risk_score": 0.1,
                    "allow_level": "auto",
                    "category": "builtin",
                },
                {
                    "id": "refactor",
                    "name": "Refactor Assistant",
                    "description": "Suggest and apply safe refactors.",
                    "scope": "repo",
                    "active": True,
                    "version": "1.0.0",
                    "risk_score": 0.2,
                    "allow_level": "review",
                    "category": "builtin",
                },
                {
                    "id": "test-gen",
                    "name": "Test Generator",
                    "description": "Generate unit tests from code.",
                    "scope": "repo",
                    "active": False,
                    "version": "0.9.0",
                    "risk_score": 0.3,
                    "allow_level": "review",
                    "category": "builtin",
                },
            ]
        self._send_json(200, {"skills": skills})

    def _handle_skill_detail(self) -> None:
        parts = [p for p in self.path.split("/") if p]
        if len(parts) < 3:
            self._send_json(400, {"error": "Missing skill ID"})
            return
        skill_id = parts[2]

        catalog = getattr(self.server, "skill_catalog", None)
        if catalog is not None and hasattr(catalog, "get"):
            try:
                entry = catalog.get(skill_id)
                if entry is None:
                    self._send_json(404, {"error": f"Skill {skill_id!r} not found"})
                    return
                self._send_json(
                    200,
                    {
                        "id": getattr(entry, "skill_id", skill_id),
                        "name": getattr(entry, "name", skill_id),
                        "description": getattr(entry, "description", ""),
                        "instructions": getattr(entry, "instructions", ""),
                        "scope": getattr(entry, "scope", "repo"),
                        "scripts": list(getattr(entry, "scripts", [])),
                        "resources": list(getattr(entry, "resources", [])),
                        "version": getattr(entry, "version", None),
                        "active": getattr(entry, "active", False),
                        "risk_score": getattr(entry, "risk_score", 0.0),
                        "allow_level": getattr(entry, "allow_level", "review"),
                        "metadata": getattr(entry, "metadata", {}),
                    },
                )
                return
            except Exception:
                logger.warning("Skill detail: failed to lookup", exc_info=True)

        self._send_json(
            200,
            {
                "id": skill_id,
                "name": skill_id.replace("-", " ").title(),
                "description": f"Skill {skill_id} description.",
                "instructions": f"# {skill_id}\n\nRun this skill with {{variable}} placeholders.",
                "scope": "repo",
                "scripts": [],
                "resources": [],
                "version": "1.0.0",
                "active": True,
                "risk_score": 0.1,
                "allow_level": "auto",
                "metadata": {},
            },
        )

    def _handle_skill_execute(self) -> None:
        try:
            payload = self._parse_json_body()
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})
            return

        skill_id = payload.get("skill_id", "")
        variables = payload.get("variables", {})
        if not isinstance(skill_id, str) or not skill_id.strip():
            self._send_json(400, {"error": "skill_id is required"})
            return

        executor = getattr(self.server, "skill_executor", None) or SkillExecutor()
        catalog = getattr(self.server, "skill_catalog", None)

        instructions = ""
        if catalog is not None and hasattr(catalog, "get"):
            try:
                entry = catalog.get(skill_id)
                if entry is not None:
                    instructions = getattr(entry, "instructions", "")
            except Exception:  # noqa: S110
                pass

        if not instructions:
            instructions = f"# {skill_id}\n\nExecuting with variables: {variables}"

        try:
            ctx = SkillContext(variables=dict(variables) if isinstance(variables, dict) else {})
            result: ExecutionResult = executor.execute(skill_id, instructions, ctx)
            self._send_json(
                200,
                {
                    "success": result.success,
                    "output": result.output,
                    "duration_ms": result.duration_ms,
                },
            )
        except Exception as exc:
            logger.exception("Skill execution error")
            self._send_json(500, {"error": str(exc), "success": False})

    def _handle_plugins(self) -> None:
        loader = getattr(self.server, "plugin_loader", None)
        plugins: list[dict[str, Any]] = []
        if loader is not None and hasattr(loader, "list_loaded"):
            try:
                for pid in loader.list_loaded():
                    plugins.append({"id": pid})
            except Exception:
                logger.warning("Plugins: failed to enumerate", exc_info=True)
        else:
            plugins = [{"id": "mcp-core"}]
        self._send_json(200, {"plugins": plugins})

    def _handle_workflows(self) -> None:
        monitor = getattr(self.server, "workflow_monitor", None)
        workflows: list[dict[str, Any]] = []
        if monitor is not None:
            try:
                monitor.summary()
                for wf_id, status in monitor._workflows.items():
                    name = monitor._names.get(wf_id, wf_id)
                    events = monitor._events.get(wf_id, [])
                    last_ts = events[-1].timestamp if events else 0.0
                    duration = 0.0
                    if len(events) >= 2 and events[0].event_type.name == "STARTED":
                        duration = last_ts - events[0].timestamp
                    workflows.append(
                        {
                            "id": wf_id,
                            "name": name,
                            "status": status.value,
                            "last_run": last_ts,
                            "duration": round(duration, 2),
                            "event_count": len(events),
                        }
                    )
            except Exception:
                logger.warning("Workflows: failed to enumerate", exc_info=True)
        else:
            workflows = [
                {
                    "id": "daily-backup",
                    "name": "Daily Backup",
                    "status": "completed",
                    "last_run": time.time() - 300,
                    "duration": 42.0,
                    "event_count": 3,
                },
                {
                    "id": "data-ingest",
                    "name": "Data Ingest",
                    "status": "failed",
                    "last_run": time.time() - 3600,
                    "duration": 0.0,
                    "event_count": 2,
                },
                {
                    "id": "health-check",
                    "name": "Health Check",
                    "status": "running",
                    "last_run": time.time() - 12,
                    "duration": 12.0,
                    "event_count": 1,
                },
            ]
        self._send_json(
            200,
            {
                "workflows": workflows,
                "summary": monitor.summary()
                if monitor
                else {"total": len(workflows), "running": 1, "completed": 1, "failed": 1},
            },
        )

    def _handle_workflow_detail(self) -> None:
        parts = [p for p in self.path.split("/") if p]
        if len(parts) < 3:
            self._send_json(400, {"error": "Missing workflow ID"})
            return
        wf_id = parts[2]
        monitor = getattr(self.server, "workflow_monitor", None)
        if monitor is not None:
            try:
                status = monitor.get_status(wf_id)
                events = monitor.get_events(wf_id)
                name = monitor._names.get(wf_id, wf_id)
                if status is None:
                    self._send_json(404, {"error": f"Workflow {wf_id!r} not found"})
                    return
                self._send_json(
                    200,
                    {
                        "id": wf_id,
                        "name": name,
                        "status": status.value,
                        "events": [
                            {
                                "type": e.event_type.value,
                                "message": e.message,
                                "timestamp": e.timestamp,
                            }
                            for e in events
                        ],
                    },
                )
                return
            except Exception:
                logger.warning("Workflow detail: failed to lookup", exc_info=True)
        self._send_json(
            200,
            {
                "id": wf_id,
                "name": wf_id.replace("-", " ").title(),
                "status": "idle",
                "events": [],
            },
        )

    def _handle_workflow_trigger(self) -> None:
        parts = [p for p in self.path.split("/") if p]
        if len(parts) < 3:
            self._send_json(400, {"error": "Missing workflow ID"})
            return
        wf_id = parts[2]
        try:
            payload = self._parse_json_body()
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})
            return
        trigger = payload.get("trigger", "")
        if not isinstance(trigger, str) or not trigger.strip():
            self._send_json(400, {"error": "trigger is required"})
            return
        monitor = getattr(self.server, "workflow_monitor", None)
        engine = getattr(self.server, "workflow_engine", None)
        if monitor is not None and engine is not None:
            try:
                # Find or create context
                registry = getattr(self.server, "workflow_registry", {})
                ctx = registry.get(wf_id)
                if ctx is None:
                    from src.workflow.workflow_engine import WorkflowContext

                    ctx = WorkflowContext(workflow_id=wf_id)
                    registry[wf_id] = ctx
                if trigger == "start":
                    monitor.start_workflow(wf_id, monitor._names.get(wf_id, wf_id))
                elif trigger == "complete":
                    monitor.complete_workflow(wf_id, "Completed via API")
                elif trigger == "fail":
                    monitor.fail_workflow(wf_id, "Failed via API")
                elif trigger == "reset":
                    monitor._workflows[wf_id] = WorkflowStatus.PENDING
                    monitor.log_event(
                        wf_id,
                        monitor._events.get(wf_id, []) and monitor._events[wf_id][0].event_type,
                        "Reset via API",
                    )
                success = engine.trigger(ctx, trigger)
                self._send_json(
                    200,
                    {
                        "success": success,
                        "workflow_id": wf_id,
                        "trigger": trigger,
                        "state": ctx.state.value,
                    },
                )
                return
            except Exception as exc:
                logger.exception("Workflow trigger error")
                self._send_json(500, {"error": str(exc), "success": False})
                return
        # Fallback: just acknowledge
        self._send_json(
            200,
            {"success": True, "workflow_id": wf_id, "trigger": trigger, "state": "idle"},
        )

    def _handle_memory(self) -> None:
        memory = getattr(self.server, "memory", None)
        if memory is not None and hasattr(memory, "dump_layer"):
            try:
                layers = {}
                for layer_name in [
                    "L0 Meta Rules",
                    "L1 Insight Index",
                    "L2 Global Facts",
                    "L3 Task Skills",
                    "L4 Session Archive",
                ]:
                    try:
                        entries = memory.dump_layer(layer_name)
                        layers[layer_name] = len(entries)
                    except Exception:
                        layers[layer_name] = 0
                self._send_json(200, {"layers": layers})
                return
            except Exception:  # noqa: S110
                pass
        self._send_json(200, {"layers": {}})

    def _handle_memory_entries(self) -> None:
        memory = getattr(self.server, "memory", None)
        if memory is not None and hasattr(memory, "dump_layer"):
            try:
                from urllib.parse import parse_qs, urlparse

                query = parse_qs(urlparse(self.path).query)
                layer_filter = query.get("layer", [None])[0]
                search_query = query.get("q", [None])[0]
                limit = int(query.get("limit", ["50"])[0])

                all_entries = []
                layer_names = [
                    "L0 Meta Rules",
                    "L1 Insight Index",
                    "L2 Global Facts",
                    "L3 Task Skills",
                    "L4 Session Archive",
                ]
                for layer_name in layer_names:
                    if layer_filter and layer_name != layer_filter:
                        continue
                    try:
                        entries = memory.dump_layer(layer_name)
                        for e in entries:
                            entry_dict = {
                                "id": getattr(e, "entry_id", "unknown"),
                                "content": getattr(e, "content", ""),
                                "layer": layer_name,
                                "timestamp": getattr(e, "timestamp", None),
                                "access_count": getattr(e, "access_count", 0),
                                "importance_score": getattr(e, "importance_score", 0.5),
                            }
                            if entry_dict.get("timestamp") is not None and hasattr(
                                entry_dict["timestamp"], "isoformat"
                            ):
                                entry_dict["timestamp"] = entry_dict["timestamp"].isoformat()
                            all_entries.append(entry_dict)
                    except Exception:  # noqa: S110
                        pass

                if search_query:
                    sq = search_query.lower()
                    all_entries = [e for e in all_entries if sq in e.get("content", "").lower()]

                all_entries.sort(key=lambda e: e.get("timestamp") or "", reverse=True)
                all_entries = all_entries[:limit]
                self._send_json(200, {"entries": all_entries})
                return
            except Exception:  # noqa: S110
                pass
        self._send_json(200, {"entries": []})

    def _handle_notification_preferences_get(self) -> None:
        prefs = getattr(self.server, "notification_preferences", {})
        self._send_json(200, {"preferences": prefs})

    def _handle_notification_preferences_post(self) -> None:
        try:
            payload = self._parse_json_body()
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})
            return
        prefs = payload.get("preferences", {})
        self.server.notification_preferences = prefs
        self._send_json(200, {"success": True})

    def _handle_logs(self) -> None:
        from urllib.parse import parse_qs, urlparse
        query = parse_qs(urlparse(self.path).query)
        level_filter = query.get("level", [None])[0]
        search = query.get("q", [None])[0]
        limit = int(query.get("limit", ["100"])[0])

        records = list(_log_buffer.records)
        if level_filter:
            records = [r for r in records if r.get("level", "").lower() == level_filter.lower()]
        if search:
            sq = search.lower()
            records = [r for r in records if sq in r.get("message", "").lower() or sq in r.get("logger", "").lower()]  # noqa: E501
        records = records[-limit:]
        self._send_json(200, {"entries": records})

    def _handle_license_validate(self) -> None:
        server = self.server
        license_key = getattr(server, "_license_key", "")
        activated = getattr(server, "_license_activated", False)
        self._send_json(200, {
            "valid": activated and bool(license_key),
            "activated": activated,
            "tier": getattr(server, "_license_tier", "trial"),
        })

    def _handle_license_activate(self) -> None:
        try:
            payload = self._parse_json_body()
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})
            return
        key = payload.get("license_key", "")
        if not key:
            self._send_json(400, {"error": "license_key required"})
            return
        # Simple validation: keys start with AURELIUS- and are 32+ chars
        if key.startswith("AURELIUS-") and len(key) >= 32:
            self.server._license_key = key
            self.server._license_activated = True
            self.server._license_tier = payload.get("tier", "pro")
            self.server.runtime_config["require_auth"] = True
            self.server.runtime_config["api_key"] = key[-16:]
            self._send_json(200, {"success": True, "tier": self.server._license_tier})
        else:
            self._send_json(403, {"error": "Invalid license key"})

    def _handle_modes(self) -> None:
        from src.agent.agent_mode_registry import AGENT_MODE_REGISTRY

        modes = []
        for registry in AGENT_MODE_REGISTRY.values():
            for mode in registry.list_modes():
                modes.append(
                    {
                        "id": mode.mode_id,
                        "name": mode.name,
                        "description": mode.description,
                        "allowed_tools": mode.allowed_tools,
                        "response_style": mode.response_style,
                    }
                )
        self._send_json(200, {"modes": modes})

    def _handle_config_get(self) -> None:
        config = getattr(self.server, "runtime_config", {})
        self._send_json(200, {"config": dict(config)})

    def _handle_config_post(self) -> None:
        try:
            payload = self._parse_json_body()
        except (ValueError, json.JSONDecodeError) as exc:
            self._send_json(400, {"error": str(exc)})
            return
        updates = payload.get("config", {})
        if not isinstance(updates, dict):
            self._send_json(400, {"error": "config must be an object"})
            return
        config = getattr(self.server, "runtime_config", {})
        for key, value in updates.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                config[key] = value
        self._send_json(200, {"success": True, "config": dict(config)})

    # ------------------------------------------------------------------
    # SSE (Server-Sent Events)
    # ------------------------------------------------------------------

    def _handle_sse(self) -> None:
        """Establish an SSE connection for real-time notifications."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        hermes = getattr(self.server, "hermes", None)
        if hermes is None:
            self.wfile.write(b"event: system\ndata: {}\n\n")
            return

        # Send a heartbeat every 15 seconds and forward notifications
        last_heartbeat = time.time()
        received_ids: set[str] = set()

        def _on_notification(n: Any) -> None:
            if n.notification_id in received_ids:
                return
            received_ids.add(n.notification_id)
            data = json.dumps(
                {
                    "id": n.notification_id,
                    "timestamp": n.timestamp,
                    "priority": n.priority,
                    "category": n.category,
                    "title": n.title,
                    "body": n.body,
                }
            )
            try:
                self.wfile.write(f"event: notification\ndata: {data}\n\n".encode())
                self.wfile.flush()
            except Exception:  # noqa: S110
                pass

        unsub = hermes.subscribe(_on_notification)

        try:
            while True:
                time.sleep(0.5)
                now = time.time()
                if now - last_heartbeat > 15:
                    self.wfile.write(b"event: heartbeat\ndata: {}\n\n")
                    self.wfile.flush()
                    last_heartbeat = now
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            unsub()


class AureliusServer(HTTPServer):
    def __init__(
        self,
        host: str,
        port: int,
        *,
        nl_parser: NLCommandParser | None = None,
        command_dispatcher: CommandDispatcher | None = None,
        skill_catalog: Any | None = None,
        plugin_loader: Any | None = None,
        supervisor: Any | None = None,
        memory: Any | None = None,
        hermes: Any | None = None,
        workflow_monitor: WorkflowMonitor | None = None,
        usage_pipeline: Any | None = None,
        bind_and_activate: bool = True,
    ):
        super().__init__((host, port), AureliusHandler, bind_and_activate=bind_and_activate)
        self.nl_parser = nl_parser
        self.command_dispatcher = command_dispatcher
        self.skill_catalog = skill_catalog
        self.plugin_loader = plugin_loader
        self.supervisor = supervisor
        self.memory = memory
        self.hermes = hermes
        self.workflow_monitor = workflow_monitor
        self.usage_pipeline = usage_pipeline
        self.activity_log = ActivityLog()
        self.workflow_registry: dict[str, Any] = {}
        self.runtime_config: dict[str, Any] = {
            "agent_mode": "supervised",
            "log_level": "info",
            "api_endpoint": "http://localhost:8080",
            "require_auth": False,
            "audit_logging": True,
            "auto_lock": False,
            "api_key": "",
        }
        self._license_key: str = ""
        self._license_activated: bool = False
        self._license_tier: str = "trial"
        self._valid_sessions: set[str] = set()


def create_aurelius_server(
    host: str,
    port: int,
    *,
    nl_parser: NLCommandParser | None = None,
    command_dispatcher: CommandDispatcher | None = None,
    skill_catalog: Any | None = None,
    plugin_loader: Any | None = None,
    supervisor: Any | None = None,
    memory: Any | None = None,
    hermes: Any | None = None,
    workflow_monitor: WorkflowMonitor | None = None,
    usage_pipeline: Any | None = None,
    bind_and_activate: bool = True,
) -> AureliusServer:
    return AureliusServer(
        host,
        port,
        nl_parser=nl_parser,
        command_dispatcher=command_dispatcher,
        skill_catalog=skill_catalog,
        plugin_loader=plugin_loader,
        supervisor=supervisor,
        memory=memory,
        hermes=hermes,
        workflow_monitor=workflow_monitor,
        usage_pipeline=usage_pipeline,
        bind_and_activate=bind_and_activate,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    arg_parser = argparse.ArgumentParser(description="Aurelius Unified Server")
    arg_parser.add_argument("--port", type=int, default=7870)
    arg_parser.add_argument("--host", default="0.0.0.0")  # nosec B104 — default for home network LAN access; user can override with --host  # noqa: S104
    args = arg_parser.parse_args()

    server = create_aurelius_server(args.host, args.port)
    url = f"http://localhost:{args.port}"
    logger.info("Serving Aurelius at %s", url)
    logger.info("Frontend dist: %s", _FRONTEND_DIST)
    if not _FRONTEND_DIST.exists():
        logger.warning(
            "Frontend dist not found at %s — run `npm run build` in frontend/",
            _FRONTEND_DIST,
        )
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down.")
        server.server_close()


__all__ = [
    "AureliusHandler",
    "AureliusServer",
    "create_aurelius_server",
]
