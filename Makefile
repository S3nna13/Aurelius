all: build_rust test

build_rust:
	cd rust_memory && PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo build --release 2>&1 || \
		echo "Warning: Rust build failed. Python code works without it."

test:
	python3 -m pytest tests.py -v

test-all:
	python3 -m pytest tests.py test_*.py -v

check:
	@for f in *.py; do \
		python3 -m py_compile "$$f" && echo "OK: $$f" || echo "FAIL: $$f"; \
	done
	cd rust_memory && PYO3_USE_ABI3_FORWARD_COMPATIBILITY=1 cargo check 2>&1 && echo "OK: rust" || echo "Rust check failed"

serve:
	python3 api_server.py --port 8000

test-api:
	python3 -c "from api_server import app, state; \
	   from api_server import load_model; \
	   load_model(device='cpu'); \
	   assert state.ready; \
	   print('API server: OK')"

clean:
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete
	cd rust_memory && cargo clean 2>/dev/null || true

count:
	@echo "Python files (LOC):"
	@wc -l *.py | tail -1
	@echo "Rust files (LOC):"
	@wc -l rust_memory/src/lib.rs rust_memory/src/checkpoint.rs | tail -1
	@echo "Test count:"
	@python3 -m pytest test_fixes.py tests.py test_*.py --collect-only -q 2>/dev/null | tail -1

.PHONY: all build_rust test check clean count
