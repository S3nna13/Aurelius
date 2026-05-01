use std::time::Instant;

use axum::{
    response::IntoResponse,
    Router,
    extract::Request,
    http::StatusCode,
    routing::any,
};

pub fn routes() -> Router<crate::app_state::AppState> {
    Router::new()
        .route("/v1/*path", any(proxy_handler))
        .route("/api/*path", any(proxy_handler))
}

async fn proxy_handler(
    axum::extract::State(state): axum::extract::State<crate::app_state::AppState>,
    req: Request,
) -> impl axum::response::IntoResponse {
    let start = Instant::now();

    let client_ip = req
        .headers()
        .get("x-forwarded-for")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("unknown");

    match state.rate_limiter.check(client_ip).await {
        crate::rate_limit::RateLimitResult::Denied { retry_after } => {
            return (
                StatusCode::TOO_MANY_REQUESTS,
                [("Retry-After", retry_after.to_string())],
                format!("{{ \"error\": \"Rate limit exceeded\", \"retry_after\": {retry_after} }}")
                    .to_string(),
            )
                .into_response();
        }
        crate::rate_limit::RateLimitResult::Allowed { remaining, reset } => {
            let _ = (remaining, reset);
        }
    }

    let req_path = req.uri().path().to_string();
    match state.proxy_client.proxy_request(req).await {
        Ok(resp) => {
            let latency = start.elapsed().as_secs_f64() * 1000.0;
            let path = req_path;
            let status = resp.status().as_u16();
            state.metrics.record_request(&path, status, latency);
            resp.into_response()
        }
        Err(status) => {
            let latency = start.elapsed().as_secs_f64() * 1000.0;
            state
                .metrics
                .record_request("/error", status.as_u16(), latency);
            (status, "{ \"error\": \"Upstream unavailable\" }".to_string()).into_response()
        }
    }
}
