#!/usr/bin/env bash
# =============================================================================
# Aurelius — Performance benchmark suite
#
# Measures:
#   - API endpoint latency (p50, p95, p99)
#   - Throughput (requests per second)
#   - WebSocket round-trip time
#   - Search index performance
#   - Data engine operation throughput
#
# Usage:
#   bash scripts/benchmark.sh                    # Run all benchmarks
#   bash scripts/benchmark.sh api               # API benchmarks only
#   bash scripts/benchmark.sh ws                # WebSocket benchmarks
#   bash scripts/benchmark.sh search            # Search benchmarks
#   bash scripts/benchmark.sh report            # Generate report only
# =============================================================================

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BASE_URL="${MIDDLE_URL:-http://localhost:3001}"
API_KEY="${AURELIUS_API_KEY:-dev-key}"
REPORT_FILE="${ROOT}/benchmark-report.md"

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

info() { echo -e "${BLUE}[bench]${NC} $1"; }
ok()   { echo -e "${GREEN}[bench]${NC} $1"; }
warn() { echo -e "${YELLOW}[bench]${NC} $1"; }
fail() { echo -e "${RED}[bench]${NC} $1"; }

RESULTS=()
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")

check_service() {
    if ! curl -sf "$BASE_URL/healthz" > /dev/null 2>&1; then
        fail "Service not reachable at $BASE_URL. Start the middle layer first."
        exit 1
    fi
    ok "Service reachable at $BASE_URL"
}

bench_api() {
    info "Benchmarking API endpoints..."

    local endpoints=(
        "/health"
        "/healthz"
        "/readyz"
        "/api/agents"
        "/api/activity"
        "/api/notifications"
        "/api/notifications/stats"
        "/api/config"
        "/api/memory/layers"
        "/api/logs"
        "/api/system"
        "/api/stats"
        "/api/search?q=test"
    )

    local header="X-API-Key: $API_KEY"
    local total=0
    local max_latency=0
    local min_latency=999999
    local all_latencies=()

    info "  Testing ${#endpoints[@]} endpoints (5 requests each)..."
    for endpoint in "${endpoints[@]}"; do
        local latencies=()
        for i in {1..5}; do
            local start=$(date +%s%N)
            local code=$(curl -s -o /dev/null -w "%{http_code}" -H "$header" "$BASE_URL$endpoint")
            local end=$(date +%s%N)
            local ms=$(( (end - start) / 1000000 ))
            latencies+=($ms)
            all_latencies+=($ms)
            total=$((total + 1))
            if [ $ms -gt $max_latency ]; then max_latency=$ms; fi
            if [ $ms -lt $min_latency ]; then min_latency=$ms; fi
        done

        # Sort and calc percentiles
        IFS=$'\n' sorted=($(sort -n <<< "${latencies[*]}")); unset IFS
        local p50="${sorted[1]}"
        local p95="${sorted[3]}"
        local p99="${sorted[4]}"
        RESULTS+=("| $endpoint | ${sorted[0]}ms | ${p50}ms | ${p95}ms | ${p99}ms | ${sorted[4]}ms | $code |")
    done

    local sorted_all=($(for l in "${all_latencies[@]}"; do echo "$l"; done | sort -n))
    local total_p50="${sorted_all[$((total / 2))]}"
    local total_p95="${sorted_all[$((total * 95 / 100))]}"
    local total_p99="${sorted_all[$((total * 99 / 100))]}"

    RESULTS+=("| **Overall** | **${min_latency}ms** | **${total_p50}ms** | **${total_p95}ms** | **${total_p99}ms** | **${max_latency}ms** | **${total} requests** |")
    ok "  API benchmarks complete ($total requests, min=${min_latency}ms, max=${max_latency}ms)"
}

bench_throughput() {
    info "Benchmarking throughput..."

    local header="X-API-Key: $API_KEY"
    local duration=5
    local concurrency=10
    local endpoint="$BASE_URL/healthz"

    info "  $concurrency concurrent connections for ${duration}s..."
    local start=$(date +%s%N)
    local count=0

    for i in $(seq 1 $concurrency); do
        (
            for j in $(seq 1 100); do
                curl -s -o /dev/null -H "$header" "$endpoint" &
            done
            wait
        ) &
    done
    wait

    # Use a quick sequential burst for measurement
    local start=$(date +%s%N)
    for i in $(seq 1 50); do
        curl -s -o /dev/null -H "$header" "$endpoint" &
    done
    wait
    local end=$(date +%s%N)
    local elapsed=$(echo "scale=2; ($end - $start) / 1000000000" | bc)
    local rps=$(echo "scale=0; 50 / $elapsed" | bc)

    RESULTS+=("| Throughput (burst) | ${rps} req/s | ${elapsed}s for 50 requests | | | | |")
    ok "  Throughput: ~${rps} req/s"
}

bench_search() {
    info "Benchmarking search..."

    local header="X-API-Key: $API_KEY"

    # Warm-up: ensure some data exists
    curl -s -H "$header" "$BASE_URL/api/search?q=system" > /dev/null

    local latencies=()
    local queries=("system" "agent" "config" "error" "memory" "notification" "test" "health" "command" "data")
    for q in "${queries[@]}"; do
        local start=$(date +%s%N)
        local resp=$(curl -s -H "$header" "$BASE_URL/api/search?q=$q&limit=5")
        local end=$(date +%s%N)
        local ms=$(( (end - start) / 1000000 ))
        latencies+=($ms)

        local count=$(echo "$resp" | python3 -c "import sys,json; d=json.load(sys.stdin); print(d.get('total', 0))" 2>/dev/null || echo "0")
        RESULTS+=("| Search: '$q' | ${ms}ms | ${count} results | | | | |")
    done

    ok "  Search benchmarks complete"
}

generate_report() {
    info "Generating benchmark report..."

    cat > "$REPORT_FILE" << EOF
# Aurelius Benchmark Report

**Date:** $TIMESTAMP
**Target:** $BASE_URL

## Results

### Endpoint Latency (ms)

| Endpoint | Min | P50 | P95 | P99 | Max | Status |
|----------|-----|-----|-----|-----|-----|--------|
$(IFS=$'\n'; echo "${RESULTS[*]}")

### Notes

- Benchmarks run against the middle layer (Node.js BFF)
- All endpoints authenticated with API key
- Results are approximate — use for relative comparison
- For production benchmarking, use a dedicated load testing tool like k6 or wrk
EOF

    ok "Report saved to $REPORT_FILE"
    cat "$REPORT_FILE"
}

main() {
    check_service

    local mode="${1:-all}"
    case "$mode" in
        all|api)
            bench_api
            bench_throughput
            bench_search
            generate_report
            ;;
        api)
            bench_api
            generate_report
            ;;
        ws)
            info "WebSocket benchmarks not yet implemented"
            ;;
        search)
            bench_search
            generate_report
            ;;
        report)
            generate_report
            ;;
        *)
            echo "Usage: $0 [all|api|ws|search|report]"
            exit 1
            ;;
    esac

    ok "Benchmarks complete!"
}

main "$@"
