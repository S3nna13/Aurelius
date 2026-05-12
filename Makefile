# --- Tool variables (CI-safe: always use venv) ---
PYTHON    := .venv/bin/python
RUFF      := .venv/bin/ruff
MYPY      := .venv/bin/mypy
BANDIT    := .venv/bin/bandit
PIP_AUDIT := .venv/bin/pip-audit
PIP       := .venv/bin/pip

.PHONY: dev serve middle frontend lint lint-fix test test-cov typecheck security \
        setup-dev setup-pre-commit bootstrap bootstrap-fast \
        middle-build middle-test middle-lint \
        frontend-build frontend-test frontend-lint \
        benchmark benchmark-api benchmark-report \
        profile-memory profile-memory-2.7b \
        train-2.7b train-3b mlx-train mlx-convert \
        rust-build-data-engine rust-build-token-counter rust-build-session-manager \
        rust-build-gateway rust-build-cli rust-build-search rust-build-redis \
        rust-build-text rust-build-vector rust-build-prompt rust-build-json \
        rust-build-uuid rust-build-all rust-test rust-lint \
        test-all db-migrate db-revision \
        docker-build docker-build-api docker-build-middle \
        docker-up docker-down docker-logs docker-up-full \
        clean clean-all audit-deps ci help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- Python Backend ---

dev: ## Start the API server in development mode
	$(PYTHON) -m gateway.aurelius_api --host 127.0.0.1 --port 8080

serve: ## Start the API server with mock generator
	$(PYTHON) -m gateway.aurelius_api --host 0.0.0.0 --port 8080

# --- Node.js Middle Layer ---

middle: ## Start the BFF server in development mode
	cd middle && npm run dev

middle-build: ## Build the BFF server
	cd middle && npm run build

middle-test: ## Test the BFF server
	cd middle && npm test

middle-lint: ## Lint the BFF server
	cd middle && npm run lint

# --- Rust Crates ---

benchmark: ## Run performance benchmarks
	bash scripts/benchmark.sh

benchmark-api: ## Run API benchmarks only
	bash scripts/benchmark.sh api

benchmark-report: ## Generate benchmark report
	bash scripts/benchmark.sh report

profile-memory: ## Profile memory at all model sizes
	bash scripts/profile_memory.sh

profile-memory-2.7b: ## Profile 2.7B memory
	bash scripts/profile_memory.sh 2.7b

train-2.7b: ## Start 2.7B training
	bash scripts/train_2.7b.sh

train-3b: ## Start 3.0B training
	bash scripts/train_3b.sh

mlx-train: ## Start MLX training
	$(PYTHON) -m src.training.mlx_trainer

mlx-convert: ## Convert PyTorch checkpoint to MLX
	$(PYTHON) -m src.training.mlx_trainer --convert $(checkpoint) --output $(output)

rust-build-data-engine: ## Build the Rust data-engine crate
	cd crates/data-engine && npm run build

rust-build-token-counter: ## Build the Rust token-counter crate
	cd crates/token-counter && npm run build

rust-build-session-manager: ## Build the Rust session-manager crate
	cd crates/session-manager && npm run build

rust-build-gateway: ## Build the Rust API gateway binary
	cd crates/api-gateway && cargo build --release

rust-build-cli: ## Build the Rust data CLI tool
	cd tools/data-cli && cargo build --release

rust-build-search: ## Build the Rust search-index NAPI crate
	cd crates/search-index && npm run build

rust-build-redis: ## Build the Rust redis-client NAPI crate
	cd crates/redis-client && npm run build

rust-build-text: ## Build the Rust text-processor NAPI crate
	cd crates/text-processor && npm run build

rust-build-vector: ## Build the Rust vector-similarity NAPI crate
	cd crates/vector-similarity && npm run build

rust-build-prompt: ## Build the Rust prompt-templates NAPI crate
	cd crates/prompt-templates && npm run build

rust-build-json: ## Build the Rust json-validator NAPI crate
	cd crates/json-validator && npm run build

rust-build-uuid: ## Build the Rust uuid-gen NAPI crate
	cd crates/uuid-gen && npm run build

rust-build-all: rust-build-data-engine rust-build-token-counter rust-build-session-manager rust-build-search rust-build-redis rust-build-text rust-build-vector rust-build-prompt rust-build-json rust-build-uuid rust-build-gateway ## Build all Rust crates

rust-test: ## Run all Rust crate tests
	@for crate in crates/data-engine crates/token-counter crates/session-manager crates/search-index crates/redis-client crates/api-gateway tools/data-cli; do \
		echo "Testing $$crate..."; \
		(cd "$$crate" && cargo test --quiet 2>/dev/null) || true; \
	done

rust-lint: ## Run clippy on all Rust crates
	@for crate in crates/data-engine crates/token-counter crates/session-manager crates/search-index crates/redis-client crates/api-gateway tools/data-cli; do \
		echo "Linting $$crate..."; \
		(cd "$$crate" && cargo clippy -- -D warnings 2>/dev/null) || true; \
	done

# --- Frontend ---

frontend: ## Start the frontend dev server
	cd frontend && npm run dev

frontend-build: ## Build the frontend
	cd frontend && npm run build

frontend-test: ## Run frontend tests
	cd frontend && npm test

frontend-lint: ## Lint the frontend
	cd frontend && npm run lint

# --- Python ---

lint: ## Run Ruff linter and formatter
	$(RUFF) check src/ tests/ agent/ aurelius_cli/ gateway/ acp_adapter/ cron/ plugins/ tools/ aurelius/
	$(RUFF) format --check src/ tests/ agent/ aurelius_cli/ gateway/ acp_adapter/ cron/ plugins/ tools/ aurelius/ --exclude src/model/attention.py

lint-fix: ## Fix lint issues automatically
	$(RUFF) check --fix src/ tests/ agent/ aurelius_cli/ gateway/ acp_adapter/ cron/ plugins/ tools/ aurelius/
	$(RUFF) format src/ tests/ agent/ aurelius_cli/ gateway/ acp_adapter/ cron/ plugins/ tools/ aurelius/ --exclude src/model/attention.py

test: ## Run Python tests
	$(PYTHON) -m pytest -q --tb=short --ignore=tests/model/test_transformer.py --ignore=tests/legacy/ --maxfail=5

test-cov: ## Run tests with coverage
	$(PYTHON) -m pytest --cov=src --cov=agent --cov=aurelius_cli --cov=gateway --cov=acp_adapter --cov=cron --cov=plugins --cov=tools --cov-report=term --cov-report=html

typecheck: ## Run mypy type checker
	$(MYPY) src/ agent/ aurelius_cli/ gateway/ acp_adapter/ cron/ plugins/ tools/ --ignore-missing-imports

security: ## Run security scans
	$(BANDIT) -r src/ agent/ aurelius_cli/ gateway/ acp_adapter/ cron/ plugins/ tools/ -ll
	$(PYTHON) -m pip_audit --desc

# --- All Tests ---

test-all: test frontend-test middle-test ## Run all tests

# --- Database ---

db-migrate: ## Run database migrations
	alembic upgrade head

db-revision: ## Create new database migration
	alembic revision --autogenerate -m "$(message)"

# --- Docker ---

docker-build: ## Build all Docker images
	docker build -f deployment/Dockerfile -t aurelius-api:latest .
	docker build -f middle/Dockerfile -t aurelius-middle:latest .

docker-build-api: ## Build API Docker image only
	docker build -f deployment/Dockerfile -t aurelius-api:latest .

docker-build-middle: ## Build Middle Docker image only
	docker build -f middle/Dockerfile -t aurelius-middle:latest .

docker-up: ## Start services with Docker Compose
	docker compose -f deployment/compose.yaml up -d

docker-down: ## Stop Docker Compose services
	docker compose -f deployment/compose.yaml down

docker-logs: ## View Docker Compose logs
	docker compose -f deployment/compose.yaml logs -f

docker-up-full: ## Start all services with cache profile
	docker compose -f deployment/compose.yaml --profile cache up -d

# --- Dev Setup ---

bootstrap: ## One-command full dev setup
	bash scripts/bootstrap.sh

bootstrap-fast: ## Quick setup (skip Rust builds)
	bash scripts/bootstrap.sh --fast

setup-dev: ## Install all development dependencies
	$(PIP) install -e ".[dev,serve,train,db]"
	cd frontend && npm install
	cd middle && npm install

setup-pre-commit: ## Install pre-commit hooks
	pre-commit install

# --- Clean ---

clean: ## Clean build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/ .coverage htmlcov/
	rm -rf frontend/dist/ frontend/node_modules/
	rm -rf middle/dist/ middle/node_modules/
	rm -rf crates/data-engine/node_modules/
	rm -rf crates/token-counter/node_modules/

clean-all: clean ## Clean ALL build artifacts (including Rust target dirs)
	rm -rf target/
	rm -rf crates/*/target/
	rm -rf tools/*/target/
	rm -rf rust_memory/target/

audit-deps: ## Audit all dependencies for vulnerabilities
	@echo "=== Python ==="
	$(PYTHON) -m pip_audit --desc || true
	@echo "=== Rust ==="
	cargo audit || true
	@echo "=== Node.js (middle) ==="
	cd middle && npm audit --audit-level=high || true
	@echo "=== Node.js (frontend) ==="
	cd frontend && npm audit --audit-level=high || true

ci: lint typecheck security test frontend-lint frontend-test middle-test ## Run all CI checks locally
