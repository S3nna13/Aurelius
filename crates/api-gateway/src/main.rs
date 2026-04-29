mod auth;
mod config;
mod metrics;
mod proxy;
mod rate_limit;
mod routes;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{middleware, Router};
use tokio::signal;
use tower_http::cors::{AllowOrigin, CorsLayer};
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::{filter::LevelFilter, EnvFilter};

#[tokio::main]
async fn main() {
    let log_level = "aurelius_gateway="
        .to_owned()
        + &config::Config::from_env()
            .log_level
            .parse::<LevelFilter>()
            .unwrap_or(LevelFilter::INFO)
            .to_string()
            .to_lowercase();

    tracing_subscriber::fmt()
        .with_env_filter(EnvFilter::from_default_env()
            .add_directive(log_level.parse().unwrap()))
        .init();

    let cfg = config::Config::from_env();
    let proxy_client = proxy::ProxyClient::new(&cfg.upstream_url);
    let rate_limiter = rate_limit::RateLimiter::new(cfg.rate_limit_rps, cfg.rate_limit_burst);
    let metrics = metrics::Metrics::new();
    let auth_service = Arc::new(auth::AuthService::new(&cfg.jwt_secret, cfg.jwt_expiry_hours));

    let cors = if cfg.allowed_origins.is_empty() {
        CorsLayer::new()
    } else {
        let origins: Vec<axum::http::HeaderValue> = cfg.allowed_origins
            .iter()
            .map(|o| o.parse().expect("Invalid CORS origin"))
            .collect();
        CorsLayer::new().allow_origin(AllowOrigin::list(origins))
    };

    let app = Router::new()
        .merge(routes::health::routes())
        .merge(routes::proxy::routes(proxy_client.clone(), rate_limiter.clone(), metrics.clone()))
        .layer(middleware::from_fn_with_state(
            auth_service.clone(),
            auth::auth_middleware,
        ))
        .layer(cors)
        .layer(RequestBodyLimitLayer::new(10 * 1024 * 1024))
        .layer(TraceLayer::new_for_http())
        .with_state(app_state::AppState { proxy_client, rate_limiter, metrics, cfg: cfg.clone() });

    let addr: SocketAddr = format!("{}:{}", cfg.host, cfg.port).parse().expect("Invalid address");
    tracing::info!("Gateway listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app.into_make_service_with_connect_info::<SocketAddr>())
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c().await.expect("Failed to install Ctrl+C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv().await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("Received Ctrl+C, shutting down"),
        _ = terminate => tracing::info!("Received SIGTERM, shutting down"),
    }
}

mod app_state {
    use crate::{config::Config, metrics::Metrics, proxy::ProxyClient, rate_limit::RateLimiter};

    #[derive(Clone)]
    pub struct AppState {
        pub proxy_client: ProxyClient,
        pub rate_limiter: RateLimiter,
        pub metrics: Metrics,
        pub cfg: Config,
    }
}
