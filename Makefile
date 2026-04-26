.PHONY: dev serve middle frontend lint test clean docker-up docker-build help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

# --- Python Backend ---

dev: ## Start the API server in development mode
	python -m src.serving.api_server --host 127.0.0.1 --port 8080

serve: ## Start the API server with mock generator
	python -m src.serving.api_server --host 0.0.0.0 --port 8080

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

rust-build-data-engine: ## Build the Rust data-engine crate
	cd crates/data-engine && npm run build

rust-build-token-counter: ## Build the Rust token-counter crate
	cd crates/token-counter && npm run build

rust-build-all: rust-build-data-engine rust-build-token-counter ## Build all Rust crates

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
	ruff check src/ tests/
	ruff format --check src/ tests/

lint-fix: ## Fix lint issues automatically
	ruff check --fix src/ tests/
	ruff format src/ tests/

test: ## Run Python tests
	python -m pytest --tb=short -q

test-cov: ## Run tests with coverage
	python -m pytest --cov=src --cov-report=term --cov-report=html

typecheck: ## Run mypy type checker
	mypy src/ --ignore-missing-imports

security: ## Run security scans
	bandit -r src/ -ll
	pip-audit --desc

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

setup-dev: ## Install all development dependencies
	pip install -e ".[dev,serve,train,db]"
	cd frontend && npm install
	cd middle && npm install
	cd crates/data-engine && npm install
	cd crates/token-counter && npm install
	pre-commit install

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
