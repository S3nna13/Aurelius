"""OpenAPI 3.1.0 specification for the Aurelius API server."""

from __future__ import annotations


def openapi_spec(host: str = "localhost", port: int = 8080) -> dict:
    scheme = "https" if port == 443 else "http"
    return {
        "openapi": "3.1.0",
        "info": {
            "title": "Aurelius API",
            "version": "0.1.0",
            "description": "OpenAI-compatible API for the Aurelius LLM server.",
            "contact": {"name": "Aurelius Systems"},
        },
        "servers": [{"url": f"{scheme}://{host}:{port}"}],
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health check",
                    "operationId": "health",
                    "tags": ["Monitoring"],
                    "responses": {
                        "200": {
                            "description": "Server is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string"},
                                            "version": {"type": "string"},
                                            "uptime": {"type": "number"},
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/healthz": {
                "get": {
                    "summary": "Liveness probe",
                    "operationId": "liveness",
                    "tags": ["Monitoring"],
                    "responses": {
                        "200": {"description": "Server is alive"},
                        "503": {"description": "Server is not alive"},
                    },
                }
            },
            "/readyz": {
                "get": {
                    "summary": "Readiness probe",
                    "operationId": "readiness",
                    "tags": ["Monitoring"],
                    "responses": {
                        "200": {"description": "Server is ready"},
                        "503": {"description": "Server is not ready"},
                    },
                }
            },
            "/openapi.json": {
                "get": {
                    "summary": "OpenAPI specification",
                    "operationId": "getOpenApi",
                    "tags": ["Docs"],
                    "responses": {"200": {"description": "OpenAPI spec in JSON format"}},
                }
            },
            "/v1/models": {
                "get": {
                    "summary": "List available models",
                    "operationId": "listModels",
                    "tags": ["Models"],
                    "security": [{"ApiKeyAuth": []}],
                    "responses": {
                        "200": {
                            "description": "List of models",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "object": {"type": "string"},
                                            "data": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "id": {"type": "string"},
                                                        "object": {"type": "string"},
                                                        "created": {"type": "integer"},
                                                        "owned_by": {"type": "string"},
                                                    },
                                                },
                                            },
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
            "/v1/chat/completions": {
                "post": {
                    "summary": "Create chat completion",
                    "operationId": "createChatCompletion",
                    "tags": ["Chat"],
                    "security": [{"ApiKeyAuth": []}],
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "required": ["messages"],
                                    "properties": {
                                        "model": {
                                            "type": "string",
                                            "default": "aurelius",
                                        },
                                        "messages": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "role": {
                                                        "type": "string",
                                                        "enum": [
                                                            "system",
                                                            "user",
                                                            "assistant",
                                                        ],
                                                    },
                                                    "content": {"type": "string"},
                                                },
                                            },
                                        },
                                        "temperature": {
                                            "type": "number",
                                            "default": 0.7,
                                            "minimum": 0,
                                            "maximum": 2,
                                        },
                                        "max_tokens": {
                                            "type": "integer",
                                            "default": 512,
                                            "minimum": 1,
                                            "maximum": 32768,
                                        },
                                        "stream": {
                                            "type": "boolean",
                                            "default": False,
                                        },
                                    },
                                }
                            }
                        },
                    },
                    "responses": {
                        "200": {
                            "description": "Chat completion response",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "id": {"type": "string"},
                                            "object": {"type": "string"},
                                            "created": {"type": "integer"},
                                            "model": {"type": "string"},
                                            "choices": {
                                                "type": "array",
                                                "items": {
                                                    "type": "object",
                                                    "properties": {
                                                        "index": {"type": "integer"},
                                                        "message": {
                                                            "type": "object",
                                                            "properties": {
                                                                "role": {"type": "string"},
                                                                "content": {"type": "string"},
                                                            },
                                                        },
                                                        "finish_reason": {"type": "string"},
                                                    },
                                                },
                                            },
                                            "usage": {
                                                "type": "object",
                                                "properties": {
                                                    "prompt_tokens": {"type": "integer"},
                                                    "completion_tokens": {"type": "integer"},
                                                    "total_tokens": {"type": "integer"},
                                                },
                                            },
                                        },
                                    }
                                }
                            },
                        }
                    },
                }
            },
        },
        "components": {
            "securitySchemes": {
                "ApiKeyAuth": {
                    "type": "apiKey",
                    "in": "header",
                    "name": "X-API-Key",
                    "description": "API key authentication",
                }
            }
        },
        "tags": [
            {"name": "Monitoring", "description": "Health and monitoring endpoints"},
            {"name": "Models", "description": "Model management endpoints"},
            {"name": "Chat", "description": "Chat completion endpoints"},
            {"name": "Docs", "description": "API documentation endpoints"},
        ],
    }


def render_docs_page(host: str, port: int) -> str:
    scheme = "https" if port == 443 else "http"
    spec_url = f"{scheme}://{host}:{port}/openapi.json"
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <title>Aurelius API Docs</title>
  <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui.css">
</head>
<body>
  <div id="swagger-ui"></div>
  <script src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5/swagger-ui-bundle.js"></script>
  <script>
    SwaggerUIBundle({{
      url: "{spec_url}",
      dom_id: "#swagger-ui",
      presets: [SwaggerUIBundle.presets.apis],
    }});
  </script>
</body>
</html>"""


def generate_openapi_json(host: str = "localhost", port: int = 8080) -> str:
    import json

    return json.dumps(openapi_spec(host, port), indent=2)
