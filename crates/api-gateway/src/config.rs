use std::env;

#[derive(Clone)]
pub struct Config {
    pub host: String,
    pub port: u16,
    pub upstream_url: String,
    pub jwt_secret: String,
    pub jwt_expiry_hours: i64,
    pub rate_limit_rps: u64,
    pub rate_limit_burst: u64,
    pub log_level: String,
    pub allowed_origins: Vec<String>,
}

impl Config {
    pub fn from_env() -> Self {
        let allowed_origins = env::var("ALLOWED_ORIGINS")
            .unwrap_or_else(|_| "http://localhost:3000".into())
            .split(',')
            .map(|s| s.trim().to_string())
            .filter(|s| !s.is_empty())
            .collect();

        Config {
            host: env::var("GATEWAY_HOST").unwrap_or_else(|_| "0.0.0.0".into()),
            port: env::var("GATEWAY_PORT").ok().and_then(|v| v.parse().ok()).unwrap_or(8080),
            upstream_url: env::var("UPSTREAM_URL").unwrap_or_else(|_| "http://127.0.0.1:3001".into()),
            jwt_secret: env::var("JWT_SECRET").expect("JWT_SECRET must be set"),
            jwt_expiry_hours: env::var("JWT_EXPIRY_HOURS").ok().and_then(|v| v.parse().ok()).unwrap_or(24),
            rate_limit_rps: env::var("RATE_LIMIT_RPS").ok().and_then(|v| v.parse().ok()).unwrap_or(100),
            rate_limit_burst: env::var("RATE_LIMIT_BURST").ok().and_then(|v| v.parse().ok()).unwrap_or(200),
            log_level: env::var("GATEWAY_LOG_LEVEL").unwrap_or_else(|_| "info".into()),
            allowed_origins,
        }
    }
}
