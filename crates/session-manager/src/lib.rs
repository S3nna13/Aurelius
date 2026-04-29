use std::collections::HashMap;
use std::sync::Mutex;

use napi_derive::napi;
use chrono::{DateTime, Utc};
use uuid::Uuid;

#[derive(Clone)]
#[napi(object)]
pub struct Session {
    pub id: String,
    pub user_id: String,
    pub role: String,
    pub created_at: String,
    pub expires_at: String,
    pub last_activity: String,
    pub metadata_json: String,
    pub ip_address: String,
    pub user_agent: String,
}

#[napi(object)]
pub struct SessionCreateOptions {
    pub user_id: String,
    pub role: String,
    pub ttl_seconds: Option<i64>,
    pub metadata: Option<String>,
    pub ip_address: Option<String>,
    pub user_agent: Option<String>,
}

#[napi(object)]
pub struct SessionStats {
    pub total_sessions: u32,
    pub active_sessions: u32,
    pub expired_sessions: u32,
}

#[napi(object)]
pub struct SessionValidation {
    pub valid: bool,
    pub session: Option<Session>,
    pub reason: Option<String>,
}

struct SessionEntry {
    session: Session,
    expires_at: DateTime<Utc>,
}

#[napi]
pub struct SessionManager {
    sessions: Mutex<HashMap<String, SessionEntry>>,
    default_ttl_seconds: i64,
}

#[napi]
impl SessionManager {
    #[napi(constructor)]
    pub fn new() -> Self {
        SessionManager {
            sessions: Mutex::new(HashMap::new()),
            default_ttl_seconds: 86400,
        }
    }

    #[napi]
    pub fn create_session(&self, options: SessionCreateOptions) -> Session {
        let now = Utc::now();
        let ttl = options.ttl_seconds.unwrap_or(self.default_ttl_seconds);
        let expires_at = now + chrono::Duration::seconds(ttl);

        let session = Session {
            id: Uuid::new_v4().to_string(),
            user_id: options.user_id,
            role: options.role,
            created_at: now.to_rfc3339(),
            expires_at: expires_at.to_rfc3339(),
            last_activity: now.to_rfc3339(),
            metadata_json: options.metadata.unwrap_or_default(),
            ip_address: options.ip_address.unwrap_or_default(),
            user_agent: options.user_agent.unwrap_or_default(),
        };

        let entry = SessionEntry {
            session: session.clone(),
            expires_at,
        };

        let mut sessions = self.sessions.lock().unwrap();
        self.cleanup_expired(&mut sessions);
        sessions.insert(session.id.clone(), entry);

        session
    }

    #[napi]
    pub fn get_session(&self, session_id: String) -> Option<Session> {
        let sessions = self.sessions.lock().unwrap();
        sessions.get(&session_id).and_then(|entry| {
            if Utc::now() < entry.expires_at {
                Some(entry.session.clone())
            } else {
                None
            }
        })
    }

    #[napi]
    pub fn validate_session(&self, session_id: String) -> SessionValidation {
        let sessions = self.sessions.lock().unwrap();
        match sessions.get(&session_id) {
            Some(entry) if Utc::now() < entry.expires_at => {
                SessionValidation {
                    valid: true,
                    session: Some(entry.session.clone()),
                    reason: None,
                }
            }
            Some(_) => SessionValidation {
                valid: false,
                session: None,
                reason: Some("Session expired".to_string()),
            },
            None => SessionValidation {
                valid: false,
                session: None,
                reason: Some("Session not found".to_string()),
            },
        }
    }

    #[napi]
    pub fn touch_session(&self, session_id: String) -> bool {
        let mut sessions = self.sessions.lock().unwrap();
        if let Some(entry) = sessions.get_mut(&session_id) {
            if Utc::now() < entry.expires_at {
                entry.session.last_activity = Utc::now().to_rfc3339();
                return true;
            }
        }
        false
    }

    #[napi]
    pub fn delete_session(&self, session_id: String) -> bool {
        let mut sessions = self.sessions.lock().unwrap();
        sessions.remove(&session_id).is_some()
    }

    #[napi]
    pub fn list_user_sessions(&self, user_id: String) -> Vec<Session> {
        let sessions = self.sessions.lock().unwrap();
        sessions.values()
            .filter(|e| e.session.user_id == user_id && Utc::now() < e.expires_at)
            .map(|e| e.session.clone())
            .collect()
    }

    #[napi]
    pub fn delete_user_sessions(&self, user_id: String) -> u32 {
        let mut sessions = self.sessions.lock().unwrap();
        let before = sessions.len() as u32;
        sessions.retain(|_, e| e.session.user_id != user_id);
        before - sessions.len() as u32
    }

    #[napi]
    pub fn get_stats(&self) -> SessionStats {
        let sessions = self.sessions.lock().unwrap();
        let now = Utc::now();
        let total = sessions.len() as u32;
        let active = sessions.values().filter(|e| now < e.expires_at).count() as u32;
        SessionStats {
            total_sessions: total,
            active_sessions: active,
            expired_sessions: total - active,
        }
    }

    fn cleanup_expired(&self, sessions: &mut HashMap<String, SessionEntry>) {
        let now = Utc::now();
        sessions.retain(|_, e| now < e.expires_at);
    }

    #[napi]
    pub fn cleanup(&self) -> u32 {
        let mut sessions = self.sessions.lock().unwrap();
        let before = sessions.len() as u32;
        self.cleanup_expired(&mut sessions);
        before - sessions.len() as u32
    }
}
