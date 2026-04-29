use std::net::SocketAddr;
use std::time::Instant;

use axum::{
    extract::{ConnectInfo, Request, State},
    http::StatusCode,
    response::IntoResponse,
    routing::any,
    Router,
};

use crate::{metrics::Metrics, proxy::ProxyClient, rate_limit::RateLimiter};

pub fn routes(
    _proxy: ProxyClient,
    _limiter: RateLimiter,
    _metrics: Metrics,
) -> Router<crate::app_state::AppState> {
    Router::new()
        .route("/v1/*path", any(proxy_handler))
        .route("/api/*path", any(proxy_handler))
}

async fn proxy_handler(
    State(state): State<crate::app_state::AppState>,
    ConnectInfo(addr): ConnectInfo<SocketAddr>,
    req: Request,
) -> impl axum::response::IntoResponse {
    let start = Instant::now();
    let path = req.uri().path_and_query()
        .map(|pq| pq.as_str())
        .unwrap_or(req.uri().path())
        .to_string();

    match state.rate_limiter.check_socket(&addr).await {
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
