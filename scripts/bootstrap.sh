#!/usr/bin/env bash
# =============================================================================
# Aurelius — One-command development bootstrap
#
# Installs/sets up all layers: Python backend, Node.js middle, Rust crates,
# frontend, pre-commit hooks.
#
# Usage:
#   bash scripts/bootstrap.sh              # Full setup
#   bash scripts/bootstrap.sh --fast       # Skip Rust builds
#   bash scripts/bootstrap.sh --python     # Python only
#   bash scripts/bootstrap.sh --frontend   # Frontend only
#   bash scripts/bootstrap.sh --middle     # Middle layer only
#   bash scripts/bootstrap.sh --rust       # Rust crates only
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m'

info() { echo -e "${BLUE}[bootstrap]${NC} $1"; }
ok()   { echo -e "${GREEN}[bootstrap]${NC} $1"; }
warn() { echo -e "${YELLOW}[bootstrap]${NC} $1"; }

FAST=false
DO_PYTHON=false
DO_FRONTEND=false
DO_MIDDLE=false
DO_RUST=false

if [[ $# -eq 0 ]]; then
    DO_PYTHON=true
    DO_FRONTEND=true
    DO_MIDDLE=true
    DO_RUST=true
fi

for arg in "$@"; do
    case "$arg" in
        --fast) FAST=true;;
        --python) DO_PYTHON=true;;
        --frontend) DO_FRONTEND=true;;
        --middle) DO_MIDDLE=true;;
        --rust) DO_RUST=true;;
        --all) DO_PYTHON=true; DO_FRONTEND=true; DO_MIDDLE=true; DO_RUST=true;;
    esac
done

# ---------------------------------------------------------------------------
# Python backend
# ---------------------------------------------------------------------------
if $DO_PYTHON; then
    info "Setting up Python backend..."
    if [ ! -d .venv ]; then
        python3 -m venv .venv
        ok "Created .venv"
    fi
    source .venv/bin/activate
    pip install -e ".[dev,serve,train,db]" -q
    ok "Python deps installed"
fi

# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------
if $DO_FRONTEND; then
    info "Setting up frontend..."
    cd frontend
    npm install --silent 2>/dev/null || npm install
    ok "Frontend deps installed ($(wc -l < package-lock.json) locked)"
    cd "$ROOT"
fi

# ---------------------------------------------------------------------------
# Middle layer
# ---------------------------------------------------------------------------
if $DO_MIDDLE; then
    info "Setting up middle layer..."
    cd middle
    npm install --silent 2>/dev/null || npm install
    ok "Middle deps installed"
    cd "$ROOT"
fi

# ---------------------------------------------------------------------------
# Rust crates
# ---------------------------------------------------------------------------
if $DO_RUST && ! $FAST; then
    info "Setting up Rust crates..."
    for crate in crates/data-engine crates/token-counter crates/session-manager crates/search-index crates/redis-client; do
        if [ -f "$crate/package.json" ]; then
            info "  Building $crate..."
            cd "$crate"
            npm install --silent 2>/dev/null || true
            npm run build 2>/dev/null && ok "  Built $crate" || warn "  Skipped $crate (NAPI build may need platform setup)"
            cd "$ROOT"
        fi
    done
    for tool in tools/data-cli crates/api-gateway; do
        if [ -f "$tool/Cargo.toml" ]; then
            info "  Building $tool..."
            (cd "$tool" && cargo build --release 2>/dev/null && ok "  Built $tool") || warn "  Skipped $tool"
        fi
    done
fi

# ---------------------------------------------------------------------------
# Pre-commit hooks
# ---------------------------------------------------------------------------
info "Setting up pre-commit hooks..."
pip install pre-commit -q 2>/dev/null || true
pre-commit install 2>/dev/null && ok "Pre-commit hooks installed" || warn "pre-commit not available"

echo ""
ok "Bootstrap complete!"
echo ""
echo "  Start development:"
echo "    make dev        Python API server"
echo "    make middle     Node.js BFF server"
echo "    make frontend   Vite dev server"
echo "    make all        All servers"
echo ""
