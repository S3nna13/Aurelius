use std::sync::Arc;

use axum::{
    extract::{Request, State},
    middleware::Next,
    response::{IntoResponse, Json, Response},
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
    jwt_expiry_hours: i64,
    api_keys: Arc<DashMap<String, Claims>>,
}

#[allow(dead_code)]
impl AuthService {
    pub fn new(secret: &str, expiry_hours: i64) -> Self {
        let api_keys = Arc::new(DashMap::new());

        AuthService {
            encoding_key: EncodingKey::from_secret(secret.as_bytes()),
            decoding_key: DecodingKey::from_secret(secret.as_bytes()),
            jwt_expiry_hours: expiry_hours,
            api_keys,
        }
    }

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

    pub fn validate_api_key(&self, key: &str) -> Option<Claims> {
        self.api_keys.get(key).map(|c| c.clone())
    }

    pub fn create_token(&self, claims: Claims) -> Result<String, jsonwebtoken::errors::Error> {
        let now = Utc::now();
        let mut claims = claims;
        claims.iat = now.timestamp() as usize;
        claims.exp = (now + Duration::hours(self.jwt_expiry_hours)).timestamp() as usize;
        encode(&Header::default(), &claims, &self.encoding_key)
    }

    pub fn validate_token(&self, token: &str) -> Result<Claims, jsonwebtoken::errors::Error> {
        let mut validation = Validation::new(jsonwebtoken::Algorithm::HS256);
        validation.validate_exp = true;
        validation.leeway = 60;
        let token_data = decode::<Claims>(token, &self.decoding_key, &validation)?;
        Ok(token_data.claims)
    }
}

#[derive(Clone)]
pub struct AuthState {
    pub auth_service: Arc<AuthService>,
    pub public_paths: Vec<&'static str>,
}

impl AuthState {
    pub fn new(auth_service: Arc<AuthService>) -> Self {
        AuthState {
            auth_service,
            public_paths: vec![
                "/health",
                "/healthz",
                "/readyz",
                "/auth/login",
                "/openapi.json",
                "/docs",
            ],
        }
    }

    fn is_public(&self, path: &str) -> bool {
        self.public_paths.iter().any(|p| path.starts_with(*p))
    }
}

pub async fn auth_middleware(
    State(state): State<AuthState>,
    req: Request,
    next: Next,
) -> Response {
    let path = req.uri().path();

    if state.is_public(path) {
        return next.run(req).await;
    }

    let auth_header = req
        .headers()
        .get("Authorization")
        .and_then(|v| v.to_str().ok());

    let token = auth_header
        .and_then(|h| h.strip_prefix("Bearer "))
        .or(auth_header);

    if let Some(token) = token
        && state.auth_service.validate_token(token).is_ok()
    {
        return next.run(req).await;
    }

    let api_key = req.headers().get("X-API-Key").and_then(|v| v.to_str().ok());

    if let Some(key) = api_key
        && state.auth_service.validate_api_key(key).is_some()
    {
        return next.run(req).await;
    }

    (
        axum::http::StatusCode::UNAUTHORIZED,
        Json(AuthError {
            error: "unauthorized".into(),
            message: "Valid authentication required".into(),
        }),
    )
        .into_response()
}
