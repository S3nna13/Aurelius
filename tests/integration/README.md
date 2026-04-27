# Aurelius Integration Tests

Cross-layer end-to-end tests verifying the full stack:
Python API → Node.js BFF → Rust engine → Frontend

## Test Structure

- `test_health.py` — Health checks across all layers
- `test_agents.py` — Agent CRUD lifecycle
- `test_activity.py` — Activity recording and querying
- `test_notifications.py` — Notification lifecycle
- `test_config.py` — Configuration management
- `test_auth.py` — Authentication flow
- `test_search.py` — Unified search across all data
- `test_websocket.py` — WebSocket messaging

## Running

```bash
# Start all services
docker compose -f deployment/compose.yaml up -d

# Run integration tests
python -m pytest tests/integration/ -v --timeout=30
```

## Test Fixtures

All tests use `conftest.py` which provides:
- `api_client` — HTTP client against the middle layer
- `auth_headers` — Pre-authenticated headers with dev key
- `engine` — Direct Rust engine connection for state verification
- `ws_client` — WebSocket client
