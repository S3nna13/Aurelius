use std::sync::Arc;

use axum::{
    Json,
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::IntoResponse,
};
use chrono::{Duration, Utc};
use dashmap::DashMap;
use jsonwebtoken::{DecodingKey, EncodingKey, Header, Validation, decode, encode};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Claims {
    pub sub: String,
    pub role: String,
    pub scopes: Vec<String>,
    pub exp: usize,
    pub iat: usize,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize)]
pub struct LoginRequest {
    pub api_key: String,
}

#[allow(dead_code)]
#[derive(Debug, Serialize, Deserialize)]
pub struct LoginResponse {
    pub token: String,
    pub token_type: String,
    pub expires_in: i64,
}

#[derive(Debug, Serialize)]
pub struct AuthError {
    pub error: String,
    pub message: String,
}

#[allow(dead_code)]
#[derive(Clone)]
pub struct AuthService {
    encoding_key: EncodingKey,
    decoding_key: DecodingKey,
    jwt_secret: Arc<String>,
    jwt_expiry_hours: i64,
    api_keys: Arc<DashMap<String, Claims>>,
}

impl AuthService {
    pub fn new(secret: &str, expiry_hours: i64) -> Self {
        let api_keys = Arc::new(DashMap::new());

        AuthService {
            encoding_key: EncodingKey::from_secret(secret.as_bytes()),
            decoding_key: DecodingKey::from_secret(secret.as_bytes()),
            jwt_secret: Arc::new(secret.to_string()),
            jwt_expiry_hours: expiry_hours,
            api_keys,
        }
    }

    #[allow(dead_code)]
    pub fn register_api_key(&self, key: &str, sub: &str, role: &str, scopes: Vec<String>) {
        self.api_keys.insert(
            key.to_string(),
            Claims {
                sub: sub.to_string(),
                role: role.to_string(),
                scopes,
                exp: 0,
                iat: 0,
            },
        );
    }

    #[allow(dead_code)]
    pub fn validate_api_key(&self, key: &str) -> Option<Claims> {
        self.api_keys.get(key).map(|c| c.clone())
    }

    #[allow(dead_code)]
    pub fn create_token(&self, claims: Claims) -> Result<String, jsonwebtoken::errors::Error> {
        let now = Utc::now();
        let mut claims = claims;
        claims.iat = now.timestamp() as usize;
        claims.exp = (now + Duration::hours(self.jwt_expiry_hours)).timestamp() as usize;
        encode(&Header::default(), &claims, &self.encoding_key)
    }

    pub fn validate_token(&self, token: &str) -> Result<Claims, jsonwebtoken::errors::Error> {
        let token_data = decode::<Claims>(token, &self.decoding_key, &Validation::default())?;
        Ok(token_data.claims)
    }
}

pub async fn auth_middleware(
    State(auth_service): State<Arc<AuthService>>,
    req: Request,
    next: Next,
) -> impl IntoResponse {
    let path = req.uri().path().to_string();

    if public_paths().contains(&path.as_str()) {
        return next.run(req).await.into_response();
    }

    let auth_header = req
        .headers()
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|h| h.to_str().ok());

    let token = match auth_header {
        Some(value) if value.to_lowercase().starts_with("bearer ") => value[7..].to_string(),
        _ => {
            return (
                StatusCode::UNAUTHORIZED,
                Json(AuthError {
                    error: "unauthorized".to_string(),
                    message: "Missing or invalid Authorization header".to_string(),
                }),
            )
                .into_response();
        }
    };

    match auth_service.validate_token(&token) {
        Ok(_claims) => next.run(req).await.into_response(),
        Err(_) => (
            StatusCode::UNAUTHORIZED,
            Json(AuthError {
                error: "unauthorized".to_string(),
                message: "Invalid or expired token".to_string(),
            }),
        )
            .into_response(),
    }
}

pub fn public_paths() -> Vec<&'static str> {
    vec![
        "/health",
        "/healthz",
        "/readyz",
        "/auth/login",
        "/openapi.json",
        "/docs",
    ]
}
