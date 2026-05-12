use std::sync::Arc;

use axum::{
    Json,
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::IntoResponse,
};
use jsonwebtoken::{DecodingKey, Validation, decode};
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Claims {
    pub sub: String,
    pub role: String,
    pub scopes: Vec<String>,
    pub exp: usize,
    pub iat: usize,
}

#[derive(Debug, Serialize)]
pub struct AuthError {
    pub error: String,
    pub message: String,
}

#[derive(Clone)]
pub struct AuthService {
    decoding_key: DecodingKey,
}

impl AuthService {
    pub fn new(secret: &str) -> Self {
        AuthService {
            decoding_key: DecodingKey::from_secret(secret.as_bytes()),
        }
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
