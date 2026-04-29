use axum::{
    body::Body,
    extract::Request,
    http::{HeaderMap, StatusCode},
    response::Response,
};
use std::sync::Arc;
use std::time::Instant;

#[derive(Clone)]
pub struct ProxyClient {
    client: reqwest::Client,
    upstream_url: Arc<String>,
}

impl ProxyClient {
    pub fn new(upstream_url: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .pool_max_idle_per_host(32)
            .build()
            .expect("Failed to create HTTP client");

        ProxyClient {
            client,
            upstream_url: Arc::new(upstream_url.trim_end_matches('/').to_string()),
        }
    }

    pub async fn proxy_request(&self, req: Request) -> Result<Response, StatusCode> {
        let path = req
            .uri()
            .path_and_query()
            .map(|pq| pq.as_str().to_string())
            .unwrap_or_else(|| req.uri().path().to_string());

        let upstream_uri = format!("{}{}", self.upstream_url, path);

        let method = req.method().clone();
        let headers = req.headers().clone();
        let body = axum::body::to_bytes(req.into_body(), 10 * 1024 * 1024)
            .await
            .map_err(|_| StatusCode::BAD_REQUEST)?;

        let mut upstream_req = self
            .client
            .request(method.clone(), &upstream_uri)
            .body(body.to_vec());

        for (key, value) in headers.iter() {
            if key.as_str() != "host" {
                upstream_req = upstream_req.header(key.clone(), value.clone());
            }
        }

        let start = Instant::now();
        let upstream_res = upstream_req.send().await.map_err(|e| {
            tracing::error!("Upstream request failed: {e}");
            StatusCode::BAD_GATEWAY
        })?;
        let latency = start.elapsed().as_secs_f64() * 1000.0;

        let status = upstream_res.status();
        let up_headers = upstream_res.headers().clone();
        let up_body = upstream_res
            .bytes()
            .await
            .map_err(|_| StatusCode::BAD_GATEWAY)?;

        let mut response_headers = HeaderMap::new();
        for (key, value) in up_headers.iter() {
            let skip_keys = ["transfer-encoding", "connection", "keep-alive"];
            if !skip_keys.contains(&key.as_str()) {
                response_headers.insert(key.clone(), value.clone());
            }
        }

        let mut response = Response::builder()
            .status(status)
            .body(Body::from(up_body.to_vec()))
            .unwrap();

        *response.headers_mut() = response_headers;

        tracing::info!(
            "{} {} -> {} ({:.1}ms)",
            method,
            path,
            status.as_u16(),
            latency
        );

        Ok(response)
    }
}
