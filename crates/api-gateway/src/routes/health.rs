use axum::{Json, Router, extract::State, routing::get};

use crate::{config::Config, metrics::Metrics, rate_limit::RateLimiter};

#[derive(serde::Serialize)]
struct HealthResponse {
    status: String,
    version: String,
    uptime_secs: f64,
    proxy: ProxyStatus,
    metrics: MetricsStatus,
}

#[derive(serde::Serialize)]
struct ProxyStatus {
    upstream: String,
    healthy: bool,
}

#[derive(serde::Serialize)]
struct MetricsStatus {
    total_requests: u64,
    requests_per_second: f64,
}

#[derive(serde::Serialize)]
struct LivenessResponse {
    alive: bool,
    timestamp: String,
}

pub fn routes() -> Router<crate::app_state::AppState> {
    Router::new()
        .route("/health", get(health_handler))
        .route("/healthz", get(liveness_handler))
        .route("/readyz", get(readiness_handler))
        .route("/metrics", get(metrics_handler))
}

async fn health_handler(
    State(state): State<crate::app_state::AppState>,
) -> Json<serde_json::Value> {
    let metrics_snap = state.metrics.snapshot();
    let rps = if metrics_snap.uptime_secs > 0.0 {
        metrics_snap.total_requests as f64 / metrics_snap.uptime_secs
    } else {
        0.0
    };

    Json(serde_json::json!({
        "status": "ok",
        "version": env!("CARGO_PKG_VERSION"),
        "uptime_secs": metrics_snap.uptime_secs,
        "proxy": {
            "upstream": state.cfg.upstream_url,
            "healthy": true,
        },
        "metrics": {
            "total_requests": metrics_snap.total_requests,
            "requests_per_second": (rps * 100.0).round() / 100.0,
            "status_counts": metrics_snap.status_counts,
            "latency_p50_ms": metrics_snap.latency_p50_ms,
            "latency_p95_ms": metrics_snap.latency_p95_ms,
            "latency_p99_ms": metrics_snap.latency_p99_ms,
        },
    }))
}

async fn liveness_handler() -> Json<serde_json::Value> {
    Json(serde_json::json!({
        "alive": true,
        "timestamp": chrono::Utc::now().to_rfc3339(),
    }))
}

async fn readiness_handler(
    State(state): State<crate::app_state::AppState>,
) -> Result<Json<serde_json::Value>, (axum::http::StatusCode, Json<serde_json::Value>)> {
    let healthy = reqwest::get(format!("{}/healthz", state.cfg.upstream_url))
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false);

    if healthy {
        Ok(Json(
            serde_json::json!({ "ready": true, "upstream": state.cfg.upstream_url }),
        ))
    } else {
        Err((
            axum::http::StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "ready": false, "upstream": state.cfg.upstream_url, "error": "Upstream not healthy"
            })),
        ))
    }
}

async fn metrics_handler(
    State(state): State<crate::app_state::AppState>,
) -> Json<serde_json::Value> {
    let snap = state.metrics.snapshot();
    Json(serde_json::json!(snap))
}
