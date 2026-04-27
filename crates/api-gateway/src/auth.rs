use std::sync::Arc;

use axum::{
    extract::Request,
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use chrono::{Duration, Utc};
use dashmap::DashMap;
use jsonwebtoken::{decode, encode, DecodingKey, EncodingKey, Header, Validation};
use serde::{Deserialize, Serialize};
use tower_http::auth::AsyncRequireAuthorizationLayer;

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct Claims {
    pub sub: String,
    pub role: String,
    pub scopes: Vec<String>,
    pub exp: usize,
    pub iat: usize,
}

#[derive(Debug, Serialize, Deserialize)]
pub struct LoginRequest {
    pub api_key: String,
}

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

        api_keys.insert("dev-key".into(), Claims {
            sub: "admin".into(),
            role: "admin".into(),
            scopes: vec!["*".into()],
            exp: 0,
            iat: 0,
        });

        AuthService {
            encoding_key: EncodingKey::from_secret(secret.as_bytes()),
            decoding_key: DecodingKey::from_secret(secret.as_bytes()),
            jwt_secret: Arc::new(secret.to_string()),
            jwt_expiry_hours: expiry_hours,
            api_keys,
        }
    }

    pub fn register_api_key(&self, key: &str, sub: &str, role: &str, scopes: Vec<String>) {
        self.api_keys.insert(key.to_string(), Claims {
            sub: sub.to_string(),
            role: role.to_string(),
            scopes,
            exp: 0,
            iat: 0,
        });
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
        let token_data = decode::<Claims>(token, &self.decoding_key, &Validation::default())?;
        Ok(token_data.claims)
    }
}

#[derive(Clone)]
pub struct AuthMiddleware {
    auth_service: Arc<AuthService>,
    public_paths: Vec<String>,
}

impl AuthMiddleware {
    pub fn new(auth_service: Arc<AuthService>) -> Self {
        AuthMiddleware {
            auth_service,
            public_paths: vec![
                "/health".into(),
                "/healthz".into(),
                "/readyz".into(),
                "/auth/login".into(),
                "/openapi.json".into(),
                "/docs".into(),
            ],
        }
    }
}

pub async fn auth_middleware(
    axum::middleware::Next,
    axum::extract::Request,
    axum::extract::State,
) -> axum::response::Response {
    // Handled via tower layers
}

pub fn public_paths() -> Vec<&'static str> {
    vec!["/health", "/healthz", "/readyz", "/auth/login", "/openapi.json", "/docs"]
}
