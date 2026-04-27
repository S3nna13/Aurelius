use std::fs;
use std::path::Path;
use std::sync::Mutex;

use serde::{Deserialize, Serialize};

use crate::engine::DataEngineInner;

#[derive(Serialize, Deserialize)]
struct EngineSnapshot {
    agents: Vec<SerializedAgent>,
    activity: Vec<SerializedActivity>,
    notifications: Vec<SerializedNotification>,
    config: std::collections::HashMap<String, String>,
    timestamp: String,
}

#[derive(Serialize, Deserialize)]
struct SerializedAgent {
    id: String,
    state: String,
    role: String,
    metrics_json: String,
}

#[derive(Serialize, Deserialize)]
struct SerializedActivity {
    id: String,
    timestamp: f64,
    command: String,
    success: bool,
    output: String,
}

#[derive(Serialize, Deserialize)]
struct SerializedNotification {
    id: String,
    timestamp: f64,
    channel: String,
    priority: String,
    category: String,
    title: String,
    body: String,
    read: bool,
    delivered: bool,
}

pub struct Persistence {
    save_path: String,
    auto_save_interval_secs: u64,
    dirty: Mutex<bool>,
}

impl Persistence {
    pub fn new(save_path: &str, auto_save_interval_secs: u64) -> Self {
        Persistence {
            save_path: save_path.to_string(),
            auto_save_interval_secs,
            dirty: Mutex::new(false),
        }
    }

    pub fn set_dirty(&self) {
        if let Ok(mut dirty) = self.dirty.lock() {
            *dirty = true;
        }
    }

    pub fn save(&self, engine: &DataEngineInner) -> Result<(), String> {
        let snapshot = EngineSnapshot {
            agents: engine.agents.iter().map(|entry| SerializedAgent {
                id: entry.key().clone(),
                state: entry.value().state.clone(),
                role: entry.value().role.clone(),
                metrics_json: entry.value().metrics_json.clone(),
            }).collect(),

            activity: engine.activity.read().unwrap().iter().map(|a| SerializedActivity {
                id: a.id.clone(),
                timestamp: a.timestamp,
                command: a.command.clone(),
                success: a.success,
                output: a.output.clone(),
            }).collect(),

            notifications: engine.notifications.read().unwrap().iter().map(|n| SerializedNotification {
                id: n.id.clone(),
                timestamp: n.timestamp,
                channel: n.channel.clone(),
                priority: n.priority.clone(),
                category: n.category.clone(),
                title: n.title.clone(),
                body: n.body.clone(),
                read: n.read,
                delivered: n.delivered,
            }).collect(),

            config: engine.config.read().unwrap().clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        let json = serde_json::to_string_pretty(&snapshot)
            .map_err(|e| format!("Serialization error: {}", e))?;

        if let Some(parent) = Path::new(&self.save_path).parent() {
            fs::create_dir_all(parent)
                .map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        fs::write(&self.save_path, &json)
            .map_err(|e| format!("Write error: {}", e))?;

        self.set_dirty();
        Ok(())
    }

    pub fn load(&self, engine: &DataEngineInner) -> Result<(), String> {
        if !Path::new(&self.save_path).exists() {
            return Err("No save file found".to_string());
        }

        let json = fs::read_to_string(&self.save_path)
            .map_err(|e| format!("Read error: {}", e))?;

        let snapshot: EngineSnapshot = serde_json::from_str(&json)
            .map_err(|e| format!("Deserialization error: {}", e))?;

        // Clear and restore agents
        engine.agents.clear();
        for agent in &snapshot.agents {
            engine.agents.insert(agent.id.clone(), crate::engine::InternalAgent {
                state: agent.state.clone(),
                role: agent.role.clone(),
                metrics_json: agent.metrics_json.clone(),
            });
        }

        // Restore activity
        {
            let mut activity = engine.activity.write().unwrap();
            activity.clear();
            for a in &snapshot.activity {
                activity.push_back(crate::engine::InternalActivity {
                    id: a.id.clone(),
                    timestamp: a.timestamp,
                    command: a.command.clone(),
                    success: a.success,
                    output: a.output.clone(),
                });
            }
        }

        // Restore notifications
        {
            let mut notifications = engine.notifications.write().unwrap();
            notifications.clear();
            for n in &snapshot.notifications {
                notifications.push_back(crate::engine::InternalNotification {
                    id: n.id.clone(),
                    timestamp: n.timestamp,
                    channel: n.channel.clone(),
                    priority: n.priority.clone(),
                    category: n.category.clone(),
                    title: n.title.clone(),
                    body: n.body.clone(),
                    read: n.read,
                    delivered: n.delivered,
                });
            }
        }

        // Restore config
        {
            let mut config = engine.config.write().unwrap();
            config.clear();
            for (key, value) in &snapshot.config {
                config.insert(key.clone(), value.clone());
            }
        }

        Ok(())
    }

    pub fn export_json(&self, engine: &DataEngineInner) -> Result<String, String> {
        let snapshot = EngineSnapshot {
            agents: engine.agents.iter().map(|entry| SerializedAgent {
                id: entry.key().clone(),
                state: entry.value().state.clone(),
                role: entry.value().role.clone(),
                metrics_json: entry.value().metrics_json.clone(),
            }).collect(),

            activity: engine.activity.read().unwrap().iter().map(|a| SerializedActivity {
                id: a.id.clone(),
                timestamp: a.timestamp,
                command: a.command.clone(),
                success: a.success,
                output: a.output.clone(),
            }).collect(),

            notifications: engine.notifications.read().unwrap().iter().map(|n| SerializedNotification {
                id: n.id.clone(),
                timestamp: n.timestamp,
                channel: n.channel.clone(),
                priority: n.priority.clone(),
                category: n.category.clone(),
                title: n.title.clone(),
                body: n.body.clone(),
                read: n.read,
                delivered: n.delivered,
            }).collect(),

            config: engine.config.read().unwrap().clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        serde_json::to_string_pretty(&snapshot)
            .map_err(|e| format!("Serialization error: {}", e))
    }

    pub fn import_json(&self, engine: &DataEngineInner, json: &str) -> Result<(), String> {
        let snapshot: EngineSnapshot = serde_json::from_str(json)
            .map_err(|e| format!("Deserialization error: {}", e))?;

        engine.agents.clear();
        for agent in &snapshot.agents {
            engine.agents.insert(agent.id.clone(), crate::engine::InternalAgent {
                state: agent.state.clone(),
                role: agent.role.clone(),
                metrics_json: agent.metrics_json.clone(),
            });
        }

        {
            let mut activity = engine.activity.write().unwrap();
            activity.clear();
            for a in &snapshot.activity {
                activity.push_back(crate::engine::InternalActivity {
                    id: a.id.clone(), timestamp: a.timestamp,
                    command: a.command.clone(), success: a.success,
                    output: a.output.clone(),
                });
            }
        }

        {
            let mut notifications = engine.notifications.write().unwrap();
            notifications.clear();
            for n in &snapshot.notifications {
                notifications.push_back(crate::engine::InternalNotification {
                    id: n.id.clone(), timestamp: n.timestamp,
                    channel: n.channel.clone(), priority: n.priority.clone(),
                    category: n.category.clone(), title: n.title.clone(),
                    body: n.body.clone(), read: n.read, delivered: n.delivered,
                });
            }
        }

        {
            let mut config = engine.config.write().unwrap();
            config.clear();
            for (key, value) in &snapshot.config {
                config.insert(key.clone(), value.clone());
            }
        }

        Ok(())
    }
}
