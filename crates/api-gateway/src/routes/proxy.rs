use std::sync::Arc;
use std::time::Instant;

use axum::{
    extract::{Request, State},
    http::StatusCode,
    response::IntoResponse,
    routing::any,
    Router,
};

use crate::{metrics::Metrics, proxy::ProxyClient, rate_limit::RateLimiter};

#[derive(Clone)]
struct ProxyState {
    proxy: ProxyClient,
    limiter: RateLimiter,
    metrics: Metrics,
}

pub fn routes(
    proxy: ProxyClient,
    limiter: RateLimiter,
    metrics: Metrics,
) -> Router<crate::app_state::AppState> {
    Router::new()
        .route("/v1/*path", any(proxy_handler))
        .route("/api/*path", any(proxy_handler))
}

async fn proxy_handler(
    State(state): State<crate::app_state::AppState>,
    req: Request,
) -> impl axum::response::IntoResponse {
    let start = Instant::now();

    let client_ip = req.headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown");

    match state.rate_limiter.check(client_ip).await {
        crate::rate_limit::RateLimitResult::Denied { retry_after } => {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                [("Retry-After", retry_after.to_string())],
                format!("{{ \"error\": \"Rate limit exceeded\", \"retry_after\": {retry_after} }}"),
            ).into_response();
        }
        crate::rate_limit::RateLimitResult::Allowed { .. } => {}
    }

    match state.proxy_client.proxy_request(req).await {
        Ok(resp) => {
            let latency = start.elapsed().as_secs_f64() * 1000.0;
            let path = resp.uri().path().to_string();
            let status = resp.status().as_u16();
            state.metrics.record_request(&path, status, latency);
            resp.into_response()
        }
        Err(status) => {
            let latency = start.elapsed().as_secs_f64() * 1000.0;
            state.metrics.record_request("/error", status.as_u16(), latency);
            (status, format!("{{ \"error\": \"Upstream unavailable\" }}")).into_response()
        }
    }
}
