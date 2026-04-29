use std::collections::VecDeque;
use std::sync::RwLock;

use chrono::Utc;
use dashmap::DashMap;
use ulid::Ulid;

pub const MAX_ACTIVITY: usize = 1000;
pub const MAX_LOGS: usize = 500;
pub const MAX_NOTIFICATIONS: usize = 500;

#[derive(Clone)]
pub struct InternalAgent {
    pub state: String,
    pub role: String,
    pub metrics_json: String,
}

#[derive(Clone)]
pub struct InternalActivity {
    pub id: String,
    pub timestamp: f64,
    pub command: String,
    pub success: bool,
    pub output: String,
}

#[derive(Clone)]
pub struct InternalNotification {
    pub id: String,
    pub timestamp: f64,
    pub channel: String,
    pub priority: String,
    pub category: String,
    pub title: String,
    pub body: String,
    pub read: bool,
    pub delivered: bool,
}

#[derive(Clone)]
pub struct InternalMemoryEntry {
    pub id: String,
    pub content: String,
    pub layer: String,
    pub timestamp: String,
    pub access_count: u32,
    pub importance_score: f64,
}

#[derive(Clone)]
pub struct InternalLog {
    pub timestamp: String,
    pub level: String,
    pub logger: String,
    pub message: String,
}

#[derive(Clone)]
pub struct InternalSkill {
    pub id: String,
    pub name: String,
    pub description: String,
    pub active: bool,
    pub version: String,
    pub category: String,
    pub risk_score: f64,
    pub allow_level: String,
    pub instructions: String,
    pub source: String,
}

#[derive(Clone)]
pub struct InternalWorkflow {
    pub id: String,
    pub name: String,
    pub status: String,
    pub last_run: f64,
    pub duration: f64,
    pub event_count: u32,
    pub source: String,
}

#[derive(Clone)]
pub struct InternalModelInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub path: String,
    pub parameter_count: i64,
    pub state: String,
    pub loaded_at: Option<String>,
    pub source: String,
}

#[derive(Clone)]
pub struct InternalDataPoint {
    pub step: u32,
    pub train_loss: f64,
    pub val_loss: f64,
    pub learning_rate: f64,
    pub accuracy: f64,
    pub grad_norm: f64,
}

#[derive(Clone)]
pub struct InternalTrainingRun {
    pub id: String,
    pub name: String,
    pub model_id: String,
    pub status: String,
    pub started_at: f64,
    pub current_epoch: u32,
    pub total_epochs: u32,
    pub best_val_loss: f64,
    pub current_lr: f64,
    pub total_steps: u32,
    pub data_points: Vec<InternalDataPoint>,
    pub source: String,
}

#[allow(dead_code)]
#[derive(Clone)]
pub struct InternalTrainingMetricsEntry {
    pub id: String,
    pub title: String,
    pub run_type: String,
    pub model_name: String,
    pub status: String,
    pub started_at: f64,
}

#[allow(dead_code)]
pub struct DataEngineInner {
    pub agents: DashMap<String, InternalAgent>,
    pub activity: RwLock<VecDeque<InternalActivity>>,
    pub notifications: RwLock<VecDeque<InternalNotification>>,
    pub notif_prefs: RwLock<std::collections::HashMap<String, bool>>,
    pub memory_layers: RwLock<std::collections::HashMap<String, VecDeque<InternalMemoryEntry>>>,
    pub config: RwLock<std::collections::HashMap<String, String>>,
    pub logs: RwLock<VecDeque<InternalLog>>,
    pub skills: RwLock<VecDeque<InternalSkill>>,
    pub workflows: RwLock<VecDeque<InternalWorkflow>>,
    pub models: RwLock<VecDeque<InternalModelInfo>>,
    pub training_runs: RwLock<Vec<InternalTrainingRun>>,
    demo_mode: bool,
}

impl DataEngineInner {
    pub fn new() -> Self {
        Self::new_with_demo_mode(Self::demo_mode_from_env())
    }

    pub fn new_with_demo_mode(demo_mode: bool) -> Self {
        let mut memory = std::collections::HashMap::new();
        memory.insert("L0 Meta Rules".to_string(), VecDeque::new());
        memory.insert("L1 Insight Index".to_string(), VecDeque::new());
        memory.insert("L2 Global Facts".to_string(), VecDeque::new());
        memory.insert("L3 Task Skills".to_string(), VecDeque::new());
        memory.insert("L4 Session Archive".to_string(), VecDeque::new());

        let mut config = std::collections::HashMap::new();
        config.insert("agent_mode".to_string(), "default".to_string());
        config.insert("log_level".to_string(), "info".to_string());
        config.insert(
            "api_endpoint".to_string(),
            "http://localhost:8080".to_string(),
        );
        config.insert("require_auth".to_string(), "false".to_string());
        config.insert("audit_logging".to_string(), "true".to_string());
        config.insert(
            "engine_mode".to_string(),
            if demo_mode {
                "demo".to_string()
            } else {
                "production".to_string()
            },
        );

        let mut skills = VecDeque::new();
        let mut workflows = VecDeque::new();
        let mut models = VecDeque::new();
        let mut training_runs = Vec::new();

        if demo_mode {
            skills.push_back(InternalSkill {
                id: "code-review".into(),
                name: "Code Review".into(),
                description: "Automated code review.".into(),
                active: true,
                version: "1.0.0".into(),
                category: "builtin".into(),
                risk_score: 0.1,
                allow_level: "auto".into(),
                instructions: "Review code for quality.".into(),
                source: "demo".into(),
            });
            skills.push_back(InternalSkill {
                id: "refactor".into(),
                name: "Refactor Assistant".into(),
                description: "Suggest safe refactors.".into(),
                active: true,
                version: "1.0.0".into(),
                category: "builtin".into(),
                risk_score: 0.2,
                allow_level: "review".into(),
                instructions: "Suggest refactors.".into(),
                source: "demo".into(),
            });
            skills.push_back(InternalSkill {
                id: "test-gen".into(),
                name: "Test Generator".into(),
                description: "Generate unit tests.".into(),
                active: false,
                version: "0.9.0".into(),
                category: "builtin".into(),
                risk_score: 0.3,
                allow_level: "review".into(),
                instructions: "Generate tests.".into(),
                source: "demo".into(),
            });

            workflows.push_back(InternalWorkflow {
                id: "daily-backup".into(),
                name: "Daily Backup".into(),
                status: "completed".into(),
                last_run: 0.0,
                duration: 42.0,
                event_count: 3,
                source: "demo".into(),
            });
            workflows.push_back(InternalWorkflow {
                id: "data-ingest".into(),
                name: "Data Ingest".into(),
                status: "failed".into(),
                last_run: 0.0,
                duration: 0.0,
                event_count: 2,
                source: "demo".into(),
            });
            workflows.push_back(InternalWorkflow {
                id: "health-check".into(),
                name: "Health Check".into(),
                status: "running".into(),
                last_run: 0.0,
                duration: 12.0,
                event_count: 1,
                source: "demo".into(),
            });

            models.push_back(InternalModelInfo {
                id: "aurelius-v1".into(),
                name: "Aurelius v1".into(),
                description: "1.3B base model".into(),
                path: "/models/aurelius-v1".into(),
                parameter_count: 1_300_000_000,
                state: "unloaded".into(),
                loaded_at: None,
                source: "demo".into(),
            });
            models.push_back(InternalModelInfo {
                id: "aurelius-chat".into(),
                name: "Aurelius Chat".into(),
                description: "1.3B chat-tuned".into(),
                path: "/models/aurelius-chat".into(),
                parameter_count: 1_300_000_000,
                state: "unloaded".into(),
                loaded_at: None,
                source: "demo".into(),
            });
            models.push_back(InternalModelInfo {
                id: "aurelius-code".into(),
                name: "Aurelius Code".into(),
                description: "1.3B code model".into(),
                path: "/models/aurelius-code".into(),
                parameter_count: 1_300_000_000,
                state: "unloaded".into(),
                loaded_at: None,
                source: "demo".into(),
            });

            let start_ts = Utc::now().timestamp() as f64 - 86400.0;
            let mut points = Vec::with_capacity(50);
            let mut val_loss = 4.2;
            for step in 0..50 {
                let train_loss = val_loss + (Self::rand_simple(step) * 0.15 - 0.075);
                val_loss = f64::max(val_loss - 0.08 * (1.0 - val_loss / 4.2), 1.5)
                    + (Self::rand_simple(step * 3) * 0.05 - 0.025);
                points.push(InternalDataPoint {
                    step: (step + 1) as u32,
                    train_loss: ((train_loss * 100.0) as f64).round() / 100.0,
                    val_loss: ((val_loss * 100.0) as f64).round() / 100.0,
                    learning_rate: if step < 10 {
                        1e-4 + step as f64 * 9e-5
                    } else {
                        1e-3 - (step as f64 - 10.0) * 2e-5
                    },
                    accuracy: if step > 5 {
                        f64::min((step as f64 - 5.0) * 1.8 + 10.0, 92.0)
                    } else {
                        0.0
                    },
                    grad_norm: Self::rand_simple(step * 7) * 2.0 + 1.0,
                });
            }
            training_runs.push(InternalTrainingRun {
                id: "run-v1-001".into(),
                name: "Aurelius v1 Pre-training".into(),
                model_id: "aurelius-v1".into(),
                status: "completed".into(),
                started_at: start_ts,
                current_epoch: 3,
                total_epochs: 3,
                best_val_loss: 1.52,
                current_lr: 0.0,
                total_steps: 50,
                data_points: points,
                source: "demo".into(),
            });

            let mut points2 = Vec::with_capacity(40);
            let mut val_loss2 = 3.8;
            for step in 0..40 {
                let train_loss = val_loss2 + (Self::rand_simple(step + 100) * 0.12 - 0.06);
                val_loss2 = f64::max(val_loss2 - 0.06 * (1.0 - val_loss2 / 3.8), 1.8)
                    + (Self::rand_simple(step * 3 + 100) * 0.04 - 0.02);
                points2.push(InternalDataPoint {
                    step: (step + 1) as u32,
                    train_loss: ((train_loss * 100.0) as f64).round() / 100.0,
                    val_loss: ((val_loss2 * 100.0) as f64).round() / 100.0,
                    learning_rate: if step < 8 {
                        2e-5 + step as f64 * 2.5e-5
                    } else {
                        2e-4 - (step as f64 - 8.0) * 5e-6
                    },
                    accuracy: if step > 3 {
                        f64::min((step as f64 - 3.0) * 2.2 + 5.0, 88.0)
                    } else {
                        0.0
                    },
                    grad_norm: Self::rand_simple(step * 7 + 100) * 1.5 + 0.8,
                });
            }
            training_runs.push(InternalTrainingRun {
                id: "run-chat-002".into(),
                name: "Aurelius Chat SFT".into(),
                model_id: "aurelius-chat".into(),
                status: "running".into(),
                started_at: start_ts + 43200.0,
                current_epoch: 2,
                total_epochs: 5,
                best_val_loss: 1.89,
                current_lr: 1.5e-4,
                total_steps: 40,
                data_points: points2,
                source: "demo".into(),
            });

            let mut points3 = Vec::with_capacity(60);
            let mut val_loss3 = 4.5;
            for step in 0..60 {
                let train_loss = val_loss3 + (Self::rand_simple(step + 200) * 0.18 - 0.09);
                val_loss3 = f64::max(val_loss3 - 0.1 * (1.0 - val_loss3 / 4.5), 1.2)
                    + (Self::rand_simple(step * 3 + 200) * 0.06 - 0.03);
                points3.push(InternalDataPoint {
                    step: (step + 1) as u32,
                    train_loss: ((train_loss * 100.0) as f64).round() / 100.0,
                    val_loss: ((val_loss3 * 100.0) as f64).round() / 100.0,
                    learning_rate: 3e-4 - step as f64 * 5e-6,
                    accuracy: if step > 8 {
                        f64::min((step as f64 - 8.0) * 1.5 + 8.0, 85.0)
                    } else {
                        0.0
                    },
                    grad_norm: Self::rand_simple(step * 7 + 200) * 2.5 + 0.5,
                });
            }
            training_runs.push(InternalTrainingRun {
                id: "run-code-003".into(),
                name: "Aurelius Code Fine-tune".into(),
                model_id: "aurelius-code".into(),
                status: "queued".into(),
                started_at: 0.0,
                current_epoch: 0,
                total_epochs: 10,
                best_val_loss: 0.0,
                current_lr: 0.0,
                total_steps: 60,
                data_points: points3,
                source: "demo".into(),
            });
        }

        DataEngineInner {
            agents: DashMap::new(),
            activity: RwLock::new(VecDeque::new()),
            notifications: RwLock::new(VecDeque::new()),
            notif_prefs: RwLock::new(std::collections::HashMap::new()),
            memory_layers: RwLock::new(memory),
            config: RwLock::new(config),
            logs: RwLock::new(VecDeque::new()),
            skills: RwLock::new(skills),
            workflows: RwLock::new(workflows),
            models: RwLock::new(models),
            training_runs: RwLock::new(training_runs),
            demo_mode,
        }
    }

    fn demo_mode_from_env() -> bool {
        let value = std::env::var("AURELIUS_DEMO_MODE").unwrap_or_default();
        matches!(
            value.trim().to_ascii_lowercase().as_str(),
            "1" | "true" | "yes" | "on"
        )
    }

    pub fn is_demo_mode(&self) -> bool {
        self.demo_mode
    }

    pub fn gen_id() -> String {
        Ulid::new().to_string()
    }

    fn rand_simple(seed: usize) -> f64 {
        ((seed * 1103515245 + 12345) & 0x7fffffff) as f64 / 2147483648.0
    }

    // -- Agents --

    pub fn get_agent(&self, id: &str) -> Option<crate::AgentState> {
        self.agents.get(id).map(|a| crate::AgentState {
            id: id.to_string(),
            state: a.state.clone(),
            role: a.role.clone(),
            metrics_json: a.metrics_json.clone(),
        })
    }

    pub fn list_agents(&self) -> Vec<crate::AgentState> {
        self.agents
            .iter()
            .map(|entry| crate::AgentState {
                id: entry.key().clone(),
                state: entry.value().state.clone(),
                role: entry.value().role.clone(),
                metrics_json: entry.value().metrics_json.clone(),
            })
            .collect()
    }

    pub fn upsert_agent(&self, id: &str, state: &str, role: &str, metrics_json: &str) {
        self.agents.insert(
            id.to_string(),
            InternalAgent {
                state: state.to_string(),
                role: role.to_string(),
                metrics_json: metrics_json.to_string(),
            },
        );
    }

    pub fn delete_agent(&self, id: &str) -> bool {
        self.agents.remove(id).is_some()
    }

    // -- Activity --

    pub fn append_activity(
        &self,
        command: &str,
        success: bool,
        output: &str,
    ) -> crate::ActivityEntry {
        let entry = crate::ActivityEntry {
            id: Self::gen_id(),
            timestamp: Utc::now().timestamp() as f64,
            command: command.to_string(),
            success,
            output: output.to_string(),
        };
        let mut log = self.activity.write().unwrap();
        log.push_back(InternalActivity {
            id: entry.id.clone(),
            timestamp: entry.timestamp,
            command: entry.command.clone(),
            success: entry.success,
            output: entry.output.clone(),
        });
        while log.len() > MAX_ACTIVITY {
            log.pop_front();
        }
        entry
    }

    pub fn get_activity(&self, limit: Option<u32>) -> Vec<crate::ActivityEntry> {
        let log = self.activity.read().unwrap();
        let limit = limit.unwrap_or(100).min(MAX_ACTIVITY as u32) as usize;
        log.iter()
            .rev()
            .take(limit)
            .map(|a| crate::ActivityEntry {
                id: a.id.clone(),
                timestamp: a.timestamp,
                command: a.command.clone(),
                success: a.success,
                output: a.output.clone(),
            })
            .collect()
    }

    pub fn search_activity(&self, query: &str, limit: Option<u32>) -> Vec<crate::ActivityEntry> {
        let log = self.activity.read().unwrap();
        let limit = limit.unwrap_or(100).min(MAX_ACTIVITY as u32) as usize;
        let q = query.to_lowercase();
        log.iter()
            .rev()
            .filter(|a| {
                a.command.to_lowercase().contains(&q) || a.output.to_lowercase().contains(&q)
            })
            .take(limit)
            .map(|a| crate::ActivityEntry {
                id: a.id.clone(),
                timestamp: a.timestamp,
                command: a.command.clone(),
                success: a.success,
                output: a.output.clone(),
            })
            .collect()
    }

    pub fn clear_activity(&self) -> u32 {
        let mut log = self.activity.write().unwrap();
        let count = log.len() as u32;
        log.clear();
        count
    }

    // -- Notifications --

    pub fn add_notification(
        &self,
        channel: &str,
        priority: &str,
        category: &str,
        title: &str,
        body: &str,
    ) -> crate::Notification {
        let n = crate::Notification {
            id: Self::gen_id(),
            timestamp: Utc::now().timestamp() as f64,
            channel: channel.to_string(),
            priority: priority.to_string(),
            category: category.to_string(),
            title: title.to_string(),
            body: body.to_string(),
            read: false,
            delivered: false,
        };
        let mut list = self.notifications.write().unwrap();
        list.push_back(InternalNotification {
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
        while list.len() > MAX_NOTIFICATIONS {
            list.pop_front();
        }
        n
    }

    pub fn get_notifications(
        &self,
        category: Option<&str>,
        priority: Option<&str>,
        read: Option<bool>,
        limit: Option<u32>,
    ) -> Vec<crate::Notification> {
        let list = self.notifications.read().unwrap();
        let limit = limit.unwrap_or(100).min(MAX_NOTIFICATIONS as u32) as usize;
        list.iter()
            .rev()
            .filter(|n| {
                if let Some(c) = category {
                    if n.category != c {
                        return false;
                    }
                }
                if let Some(p) = priority {
                    if n.priority != p {
                        return false;
                    }
                }
                if let Some(r) = read {
                    if n.read != r {
                        return false;
                    }
                }
                true
            })
            .take(limit)
            .map(|n| crate::Notification {
                id: n.id.clone(),
                timestamp: n.timestamp,
                channel: n.channel.clone(),
                priority: n.priority.clone(),
                category: n.category.clone(),
                title: n.title.clone(),
                body: n.body.clone(),
                read: n.read,
                delivered: n.delivered,
            })
            .collect()
    }

    pub fn mark_notification_read(&self, id: &str) -> bool {
        let mut list = self.notifications.write().unwrap();
        for n in list.iter_mut() {
            if n.id == id {
                n.read = true;
                return true;
            }
        }
        false
    }

    pub fn mark_all_notifications_read(&self, category: Option<&str>) -> u32 {
        let mut list = self.notifications.write().unwrap();
        let mut count = 0;
        for n in list.iter_mut() {
            if let Some(c) = category {
                if n.category != c {
                    continue;
                }
            }
            if !n.read {
                n.read = true;
                count += 1;
            }
        }
        count
    }

    pub fn get_notification_stats(&self) -> crate::NotificationStats {
        let list = self.notifications.read().unwrap();
        let total = list.len() as u32;
        let unread = list.iter().filter(|n| !n.read).count() as u32;
        crate::NotificationStats { unread, total }
    }

    pub fn clear_notifications(&self) -> u32 {
        let mut list = self.notifications.write().unwrap();
        let count = list.len() as u32;
        list.clear();
        count
    }

    // -- Memory --

    pub fn get_memory_layers(&self) -> Vec<crate::MemoryLayer> {
        let layers = self.memory_layers.read().unwrap();
        layers
            .iter()
            .map(|(name, entries)| crate::MemoryLayer {
                name: name.clone(),
                entries: entries.len() as u32,
            })
            .collect()
    }

    pub fn add_memory_entry(&self, layer_name: &str, content: &str) {
        let mut layers = self.memory_layers.write().unwrap();
        let entry = InternalMemoryEntry {
            id: Self::gen_id(),
            content: content.to_string(),
            layer: layer_name.to_string(),
            timestamp: Utc::now().to_rfc3339(),
            access_count: 0,
            importance_score: 0.5,
        };
        layers
            .entry(layer_name.to_string())
            .or_default()
            .push_back(entry);
    }

    pub fn get_memory_entries(
        &self,
        layer: Option<&str>,
        query: Option<&str>,
        limit: Option<u32>,
    ) -> Vec<crate::MemoryEntry> {
        let layers = self.memory_layers.read().unwrap();
        const MAX_LIMIT: u32 = 1000;
        let limit = limit.unwrap_or(50).min(MAX_LIMIT) as usize;
        let mut results = Vec::new();
        for (name, entries) in layers.iter() {
            if let Some(l) = layer {
                if name != l {
                    continue;
                }
            }
            for e in entries.iter() {
                if let Some(q) = query {
                    if !q.is_empty() && !e.content.to_lowercase().contains(&q.to_lowercase()) {
                        continue;
                    }
                }
                results.push(crate::MemoryEntry {
                    id: e.id.clone(),
                    content: e.content.clone(),
                    layer: e.layer.clone(),
                    timestamp: e.timestamp.clone(),
                    access_count: e.access_count,
                    importance_score: e.importance_score,
                });
            }
        }
        results.sort_by(|a, b| b.timestamp.cmp(&a.timestamp));
        results.truncate(limit);
        results
    }

    pub fn delete_memory_entry(&self, id: &str) -> bool {
        let mut layers = self.memory_layers.write().unwrap();
        for entries in layers.values_mut() {
            if let Some(pos) = entries.iter().position(|e| e.id == id) {
                entries.remove(pos);
                return true;
            }
        }
        false
    }

    // -- Config --

    pub fn get_config(&self, key: &str) -> Option<String> {
        self.config.read().unwrap().get(key).cloned()
    }

    pub fn get_all_config(&self) -> std::collections::HashMap<String, String> {
        self.config.read().unwrap().clone()
    }

    pub fn set_config(&self, key: &str, value: &str) {
        self.config
            .write()
            .unwrap()
            .insert(key.to_string(), value.to_string());
    }

    // -- Logs --

    pub fn append_log(&self, level: &str, logger: &str, message: &str) {
        let mut logs = self.logs.write().unwrap();
        logs.push_back(InternalLog {
            timestamp: Utc::now().format("%Y-%m-%dT%H:%M:%S").to_string(),
            level: level.to_string(),
            logger: logger.to_string(),
            message: message.to_string(),
        });
        while logs.len() > MAX_LOGS {
            logs.pop_front();
        }
    }

    pub fn get_logs(
        &self,
        level: Option<&str>,
        query: Option<&str>,
        limit: Option<u32>,
    ) -> Vec<crate::LogRecord> {
        let logs = self.logs.read().unwrap();
        let limit = limit.unwrap_or(100).min(MAX_LOGS as u32) as usize;
        logs.iter()
            .rev()
            .filter(|l| {
                if let Some(lv) = level {
                    if l.level.to_lowercase() != lv.to_lowercase() {
                        return false;
                    }
                }
                if let Some(q) = query {
                    if !q.is_empty()
                        && !l.message.to_lowercase().contains(&q.to_lowercase())
                        && !l.logger.to_lowercase().contains(&q.to_lowercase())
                    {
                        return false;
                    }
                }
                true
            })
            .take(limit)
            .map(|l| crate::LogRecord {
                timestamp: l.timestamp.clone(),
                level: l.level.clone(),
                logger: l.logger.clone(),
                message: l.message.clone(),
            })
            .collect()
    }

    pub fn search_logs(
        &self,
        query: &str,
        level: Option<&str>,
        limit: Option<u32>,
    ) -> Vec<crate::LogRecord> {
        let logs = self.logs.read().unwrap();
        let limit = limit.unwrap_or(100).min(MAX_LOGS as u32) as usize;
        let q = query.to_lowercase();
        logs.iter()
            .rev()
            .filter(|l| {
                if let Some(lv) = level {
                    if l.level.to_lowercase() != lv.to_lowercase() {
                        return false;
                    }
                }
                l.message.to_lowercase().contains(&q) || l.logger.to_lowercase().contains(&q)
            })
            .take(limit)
            .map(|l| crate::LogRecord {
                timestamp: l.timestamp.clone(),
                level: l.level.clone(),
                logger: l.logger.clone(),
                message: l.message.clone(),
            })
            .collect()
    }

    pub fn clear_logs(&self) -> u32 {
        let mut logs = self.logs.write().unwrap();
        let count = logs.len() as u32;
        logs.clear();
        count
    }

    // -- Skills --

    pub fn list_skills(&self) -> Vec<crate::SkillEntry> {
        self.skills
            .read()
            .unwrap()
            .iter()
            .map(|s| crate::SkillEntry {
                id: s.id.clone(),
                name: s.name.clone(),
                description: s.description.clone(),
                active: s.active,
                version: s.version.clone(),
                category: s.category.clone(),
                risk_score: s.risk_score,
                allow_level: s.allow_level.clone(),
                source: s.source.clone(),
            })
            .collect()
    }

    pub fn get_skill(&self, id: &str) -> Option<crate::SkillEntry> {
        self.skills
            .read()
            .unwrap()
            .iter()
            .find(|s| s.id == id)
            .map(|s| crate::SkillEntry {
                id: s.id.clone(),
                name: s.name.clone(),
                description: s.description.clone(),
                active: s.active,
                version: s.version.clone(),
                category: s.category.clone(),
                risk_score: s.risk_score,
                allow_level: s.allow_level.clone(),
                source: s.source.clone(),
            })
    }

    // -- Workflows --

    pub fn list_workflows(&self) -> Vec<crate::WorkflowEntry> {
        self.workflows
            .read()
            .unwrap()
            .iter()
            .map(|w| crate::WorkflowEntry {
                id: w.id.clone(),
                name: w.name.clone(),
                status: w.status.clone(),
                last_run: w.last_run,
                duration: w.duration,
                event_count: w.event_count,
                source: w.source.clone(),
            })
            .collect()
    }

    pub fn get_workflow(&self, id: &str) -> Option<crate::WorkflowEntry> {
        self.workflows
            .read()
            .unwrap()
            .iter()
            .find(|w| w.id == id)
            .map(|w| crate::WorkflowEntry {
                id: w.id.clone(),
                name: w.name.clone(),
                status: w.status.clone(),
                last_run: w.last_run,
                duration: w.duration,
                event_count: w.event_count,
                source: w.source.clone(),
            })
    }

    pub fn update_workflow_status(&self, id: &str, status: &str) -> bool {
        let mut wf = self.workflows.write().unwrap();
        for w in wf.iter_mut() {
            if w.id == id {
                w.status = status.to_string();
                w.last_run = Utc::now().timestamp() as f64;
                w.event_count += 1;
                return true;
            }
        }
        false
    }

    // -- Models --

    pub fn list_models(&self) -> Vec<crate::ModelInfo> {
        self.models
            .read()
            .unwrap()
            .iter()
            .map(|m| crate::ModelInfo {
                id: m.id.clone(),
                name: m.name.clone(),
                description: m.description.clone(),
                path: m.path.clone(),
                parameter_count: m.parameter_count,
                state: m.state.clone(),
                loaded_at: m.loaded_at.clone(),
                source: m.source.clone(),
            })
            .collect()
    }

    pub fn get_model(&self, id: &str) -> Option<crate::ModelInfo> {
        self.models
            .read()
            .unwrap()
            .iter()
            .find(|m| m.id == id)
            .map(|m| crate::ModelInfo {
                id: m.id.clone(),
                name: m.name.clone(),
                description: m.description.clone(),
                path: m.path.clone(),
                parameter_count: m.parameter_count,
                state: m.state.clone(),
                loaded_at: m.loaded_at.clone(),
                source: m.source.clone(),
            })
    }

    pub fn set_model_state(&self, id: &str, state: &str) -> bool {
        let mut models = self.models.write().unwrap();
        for m in models.iter_mut() {
            if m.id == id {
                m.state = state.to_string();
                m.loaded_at = Some(Utc::now().to_rfc3339());
                return true;
            }
        }
        false
    }

    // -- Training Runs --

    pub fn list_training_runs(&self) -> Vec<crate::TrainingRunSummary> {
        self.training_runs
            .read()
            .unwrap()
            .iter()
            .map(|r| crate::TrainingRunSummary {
                id: r.id.clone(),
                name: r.name.clone(),
                model_id: r.model_id.clone(),
                status: r.status.clone(),
                started_at: r.started_at,
                current_epoch: r.current_epoch,
                total_epochs: r.total_epochs,
                best_val_loss: r.best_val_loss,
                current_lr: r.current_lr,
                total_steps: r.total_steps,
                data_point_count: r.data_points.len() as u32,
                source: r.source.clone(),
            })
            .collect()
    }

    pub fn get_training_run(&self, id: &str) -> Option<crate::TrainingRunDetail> {
        self.training_runs
            .read()
            .unwrap()
            .iter()
            .find(|r| r.id == id)
            .map(|r| {
                let steps = r.data_points.iter().map(|p| p.step).collect::<Vec<_>>();
                let train_losses = r
                    .data_points
                    .iter()
                    .map(|p| p.train_loss)
                    .collect::<Vec<_>>();
                let val_losses = r.data_points.iter().map(|p| p.val_loss).collect::<Vec<_>>();
                let lrs = r
                    .data_points
                    .iter()
                    .map(|p| p.learning_rate)
                    .collect::<Vec<_>>();
                let accs = r.data_points.iter().map(|p| p.accuracy).collect::<Vec<_>>();
                crate::TrainingRunDetail {
                    id: r.id.clone(),
                    name: r.name.clone(),
                    model_id: r.model_id.clone(),
                    status: r.status.clone(),
                    started_at: r.started_at,
                    current_epoch: r.current_epoch,
                    total_epochs: r.total_epochs,
                    best_val_loss: r.best_val_loss,
                    current_lr: r.current_lr,
                    total_steps: r.total_steps,
                    steps,
                    train_losses,
                    val_losses,
                    learning_rates: lrs,
                    accuracies: accs,
                    source: r.source.clone(),
                }
            })
    }

    pub fn create_training_run(
        &self,
        name: &str,
        model_id: &str,
        total_epochs: u32,
    ) -> crate::TrainingRunSummary {
        let mut runs = self.training_runs.write().unwrap();
        let run = InternalTrainingRun {
            id: Self::gen_id(),
            name: name.into(),
            model_id: model_id.into(),
            status: "queued".into(),
            started_at: Utc::now().timestamp() as f64,
            current_epoch: 0,
            total_epochs,
            best_val_loss: 0.0,
            current_lr: 0.0,
            total_steps: 0,
            data_points: Vec::new(),
            source: if self.demo_mode {
                "demo".into()
            } else {
                "live".into()
            },
        };
        let summary = crate::TrainingRunSummary {
            id: run.id.clone(),
            name: run.name.clone(),
            model_id: run.model_id.clone(),
            status: run.status.clone(),
            started_at: run.started_at,
            current_epoch: run.current_epoch,
            total_epochs: run.total_epochs,
            best_val_loss: run.best_val_loss,
            current_lr: run.current_lr,
            total_steps: run.total_steps,
            data_point_count: 0,
            source: run.source.clone(),
        };
        runs.push(run);
        summary
    }

    // -- Stats --

    pub fn get_stats(&self) -> crate::SystemStats {
        let activity = self.activity.read().unwrap();
        let notifications = self.notifications.read().unwrap();
        let logs = self.logs.read().unwrap();
        let memory = self.memory_layers.read().unwrap();

        crate::SystemStats {
            agent_count: self.agents.len() as u32,
            activity_count: activity.len() as u32,
            notification_count: notifications.len() as u32,
            notification_unread: notifications.iter().filter(|n| !n.read).count() as u32,
            memory_entry_count: memory.values().map(|v| v.len() as u32).sum(),
            log_count: logs.len() as u32,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DataEngineInner;
    use crate::persistence::{Persistence, resolve_data_path};

    #[test]
    fn production_boot_starts_empty() {
        let engine = DataEngineInner::new_with_demo_mode(false);

        assert!(engine.list_skills().is_empty());
        assert!(engine.list_workflows().is_empty());
        assert!(engine.list_models().is_empty());
        assert!(engine.list_training_runs().is_empty());
        assert_eq!(
            engine.get_config("engine_mode").as_deref(),
            Some("production")
        );

        let run = engine.create_training_run("Live Run", "model-x", 4);
        assert_eq!(run.source, "live");
        assert_eq!(engine.get_training_run(&run.id).unwrap().source, "live");
    }

    #[test]
    fn demo_boot_round_trips_through_persistence() {
        let engine = DataEngineInner::new_with_demo_mode(true);
        let path = format!("{}.json", DataEngineInner::gen_id());
        let persistence = Persistence::new(&path, 0);

        persistence.save(&engine).unwrap();

        let restored = DataEngineInner::new_with_demo_mode(false);
        assert!(restored.list_skills().is_empty());
        assert!(restored.list_workflows().is_empty());
        assert!(restored.list_models().is_empty());
        assert!(restored.list_training_runs().is_empty());

        persistence.load(&restored).unwrap();

        assert_eq!(restored.get_config("engine_mode").as_deref(), Some("demo"));

        let skills = restored.list_skills();
        assert_eq!(skills.len(), 3);
        assert!(skills.iter().all(|skill| skill.source == "demo"));

        let workflows = restored.list_workflows();
        assert_eq!(workflows.len(), 3);
        assert!(workflows.iter().all(|workflow| workflow.source == "demo"));

        let models = restored.list_models();
        assert_eq!(models.len(), 3);
        assert!(models.iter().all(|model| model.source == "demo"));

        let runs = restored.list_training_runs();
        assert_eq!(runs.len(), 3);
        assert!(runs.iter().all(|run| run.source == "demo"));

        let resolved = resolve_data_path(&path).unwrap();
        let _ = std::fs::remove_file(resolved);
    }
}
