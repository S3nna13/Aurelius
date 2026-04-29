mod auth;
mod config;
mod metrics;
mod proxy;
mod rate_limit;
mod routes;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{Router, middleware};
use tokio::signal;
use tower_http::cors::CorsLayer;
use tower_http::limit::RequestBodyLimitLayer;
use tower_http::trace::TraceLayer;
use tracing_subscriber::EnvFilter;

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::from_default_env().add_directive("aurelius_gateway=info".parse().unwrap()),
        )
        .init();

    let cfg = config::Config::from_env();
    let proxy_client = proxy::ProxyClient::new(&cfg.upstream_url);
    let rate_limiter = rate_limit::RateLimiter::new(cfg.rate_limit_rps, cfg.rate_limit_burst);
    let metrics = metrics::Metrics::new();

    let jwt_secret = std::env::var("JWT_SECRET").unwrap_or_else(|_| "aurelius-dev-secret".into());
    let auth_service = Arc::new(auth::AuthService::new(&jwt_secret, 24));
    let auth_state = auth::AuthState::new(auth_service);

    let app = Router::new()
        .merge(routes::health::routes())
        .merge(routes::proxy::routes(
            proxy_client.clone(),
            rate_limiter.clone(),
            metrics.clone(),
        ))
        .layer(middleware::from_fn_with_state(
            auth_state.clone(),
            auth::auth_middleware,
        ))
        .layer(CorsLayer::permissive())
        .layer(RequestBodyLimitLayer::new(10 * 1024 * 1024))
        .layer(TraceLayer::new_for_http())
        .with_state(app_state::AppState {
            proxy_client,
            rate_limiter,
            metrics,
            cfg: cfg.clone(),
            auth_state,
        });

    let addr: SocketAddr = format!("{}:{}", cfg.host, cfg.port)
        .parse()
        .expect("Invalid address");
    tracing::info!("Gateway listening on {addr}");

    let listener = tokio::net::TcpListener::bind(addr).await.unwrap();
    axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await
        .unwrap();
}

async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };
    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };
    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => tracing::info!("Received Ctrl+C, shutting down"),
        _ = terminate => tracing::info!("Received SIGTERM, shutting down"),
    }
}

mod app_state {
    use crate::{
        auth::AuthState, config::Config, metrics::Metrics, proxy::ProxyClient,
        rate_limit::RateLimiter,
    };

    #[derive(Clone)]
    pub struct AppState {
        pub proxy_client: ProxyClient,
        pub rate_limiter: RateLimiter,
        pub metrics: Metrics,
        pub cfg: Config,
        pub auth_state: AuthState,
    }
}
