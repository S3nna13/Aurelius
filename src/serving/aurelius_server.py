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
import os
import time
import uuid
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

from src.agent.command_dispatcher import CommandDispatcher
from src.agent.nl_command_parser import NLCommandParseError, NLCommandParser
from src.serving.mission_control import ActivityLog

logger = logging.getLogger(__name__)

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

    def do_GET(self):
        # API routes
        if self.path == "/api/health":
            self._handle_health()
            return
        if self.path == "/api/status":
            self._handle_status()
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
        if self.path == "/api/plugins":
            self._handle_plugins()
            return
        if self.path == "/api/workflows":
            self._handle_workflows()
            return
        if self.path == "/api/memory":
            self._handle_memory()
            return
        if self.path == "/api/modes":
            self._handle_modes()
            return
        if self.path == "/api/events":
            self._handle_sse()
            return

        # Static files
        path = self.path
        if path == "/":
            path = "/index.html"
        self._serve_static(path)

    def do_POST(self):
        if self.path == "/api/command":
            self._handle_command()
            return
        if self.path == "/api/notifications/read":
            self._handle_notification_read()
            return
        if self.path == "/api/notifications/read-all":
            self._handle_notification_read_all()
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
                status["agents"].append(
                    {"id": aid, "state": str(getattr(rec, "state", "UNKNOWN"))}
                )
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
            except Exception:
                pass
        else:
            status["memory"] = {"total_entries": 0}

        # Hermes stats
        hermes = getattr(server, "hermes", None)
        if hermes is not None:
            status["notifications"] = {"unread": hermes.unread_count()}
        else:
            status["notifications"] = {"unread": 0}

        self._send_json(200, status)

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
        read = (
            {"true": True, "false": False}.get(read_str.lower()) if read_str else None
        )
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
                            "active": getattr(entry, "active", False),
                        }
                    )
            except Exception:
                logger.warning("Skills: failed to enumerate", exc_info=True)
        else:
            skills = [{"id": "code-review", "name": "Code Review", "active": True}]
        self._send_json(200, {"skills": skills})

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
        registry = getattr(self.server, "workflow_registry", None)
        workflows: list[dict[str, Any]] = []
        if registry is not None and isinstance(registry, dict):
            workflows = [{"id": k, "name": k} for k in registry.keys()]
        self._send_json(200, {"workflows": workflows})

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
            except Exception:
                pass
        self._send_json(200, {"layers": {}})

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
                self.wfile.write(
                    f"event: notification\ndata: {data}\n\n".encode("utf-8")
                )
                self.wfile.flush()
            except Exception:
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
        bind_and_activate: bool = True,
    ):
        super().__init__(
            (host, port), AureliusHandler, bind_and_activate=bind_and_activate
        )
        self.nl_parser = nl_parser
        self.command_dispatcher = command_dispatcher
        self.skill_catalog = skill_catalog
        self.plugin_loader = plugin_loader
        self.supervisor = supervisor
        self.memory = memory
        self.hermes = hermes
        self.activity_log = ActivityLog()
        self.workflow_registry: dict[str, Any] = {}


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
        bind_and_activate=bind_and_activate,
    )


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(name)s: %(message)s"
    )

    arg_parser = argparse.ArgumentParser(description="Aurelius Unified Server")
    arg_parser.add_argument("--port", type=int, default=7870)
    arg_parser.add_argument(
        "--host", default="0.0.0.0"
    )  # nosec B104 — default for home network LAN access; user can override with --host
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
