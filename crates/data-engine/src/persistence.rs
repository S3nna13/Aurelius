use std::collections::VecDeque;
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

const MAX_JSON_INPUT_SIZE: usize = 100 * 1024 * 1024;
const MAX_AGENTS: usize = 10_000;
const MAX_ACTIVITY_IMPORT: usize = 10_000;
const MAX_NOTIFICATIONS_IMPORT: usize = 10_000;
const MAX_CONFIG_ENTRIES: usize = 10_000;
const MAX_SKILLS_IMPORT: usize = 10_000;
const MAX_WORKFLOWS_IMPORT: usize = 10_000;
const MAX_MODELS_IMPORT: usize = 10_000;
const MAX_TRAINING_RUNS_IMPORT: usize = 10_000;
const MAX_TRAINING_POINTS_IMPORT: usize = 100_000;
const MAX_FIELD_LENGTH: usize = 10_000;

pub(crate) fn resolve_data_path(raw_path: &str) -> Result<PathBuf, String> {
    let base = std::env::current_dir()
        .map_err(|e| format!("Failed to get working directory: {}", e))?
        .join("aurelius-data");

    fs::create_dir_all(&base).map_err(|e| format!("Failed to create data directory: {}", e))?;

    let canonical_base = fs::canonicalize(&base)
        .map_err(|e| format!("Failed to canonicalize base directory: {}", e))?;

    let raw = Path::new(raw_path);
    if raw.is_absolute() {
        return Err("Absolute paths are not allowed".to_string());
    }
    if raw
        .components()
        .any(|c| matches!(c, std::path::Component::ParentDir))
    {
        return Err("Path traversal is not allowed".to_string());
    }

    let joined = canonical_base.join(raw);

    match joined.canonicalize() {
        Ok(canonical) => {
            if !canonical.starts_with(&canonical_base) {
                return Err("Path escapes data directory".to_string());
            }
            Ok(canonical)
        }
        Err(_) => {
            let parent = joined.parent().ok_or_else(|| "Invalid path".to_string())?;
            let file_name = joined
                .file_name()
                .ok_or_else(|| "Invalid path".to_string())?;
            let canonical_parent =
                fs::canonicalize(parent).map_err(|e| format!("Path resolution error: {}", e))?;
            let resolved = canonical_parent.join(file_name);
            if !resolved.starts_with(&canonical_base) {
                return Err("Path escapes data directory".to_string());
            }
            Ok(resolved)
        }
    }
}

use serde::{Deserialize, Serialize};

use crate::engine::DataEngineInner;

fn default_engine_mode() -> String {
    "production".to_string()
}

fn default_record_source() -> String {
    "legacy".to_string()
}

#[derive(Serialize, Deserialize)]
struct EngineSnapshot {
    #[serde(default)]
    agents: Vec<SerializedAgent>,
    #[serde(default)]
    activity: Vec<SerializedActivity>,
    #[serde(default)]
    notifications: Vec<SerializedNotification>,
    #[serde(default)]
    skills: Vec<SerializedSkill>,
    #[serde(default)]
    workflows: Vec<SerializedWorkflow>,
    #[serde(default)]
    models: Vec<SerializedModelInfo>,
    #[serde(default)]
    training_runs: Vec<SerializedTrainingRun>,
    #[serde(default = "default_engine_mode")]
    engine_mode: String,
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

#[derive(Serialize, Deserialize)]
struct SerializedSkill {
    id: String,
    name: String,
    description: String,
    active: bool,
    version: String,
    category: String,
    risk_score: f64,
    allow_level: String,
    instructions: String,
    #[serde(default = "default_record_source")]
    source: String,
}

#[derive(Serialize, Deserialize)]
struct SerializedWorkflow {
    id: String,
    name: String,
    status: String,
    last_run: f64,
    duration: f64,
    event_count: u32,
    #[serde(default = "default_record_source")]
    source: String,
}

#[derive(Serialize, Deserialize)]
struct SerializedModelInfo {
    id: String,
    name: String,
    description: String,
    path: String,
    parameter_count: i64,
    state: String,
    loaded_at: Option<String>,
    #[serde(default = "default_record_source")]
    source: String,
}

#[derive(Serialize, Deserialize)]
struct SerializedTrainingDataPoint {
    step: u32,
    train_loss: f64,
    val_loss: f64,
    learning_rate: f64,
    accuracy: f64,
    grad_norm: f64,
}

#[derive(Serialize, Deserialize)]
struct SerializedTrainingRun {
    id: String,
    name: String,
    model_id: String,
    status: String,
    started_at: f64,
    current_epoch: u32,
    total_epochs: u32,
    best_val_loss: f64,
    current_lr: f64,
    total_steps: u32,
    #[serde(default)]
    data_points: Vec<SerializedTrainingDataPoint>,
    #[serde(default = "default_record_source")]
    source: String,
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
        let resolved_path = resolve_data_path(&self.save_path)?;

        let snapshot = EngineSnapshot {
            agents: engine
                .agents
                .iter()
                .map(|entry| SerializedAgent {
                    id: entry.key().clone(),
                    state: entry.value().state.clone(),
                    role: entry.value().role.clone(),
                    metrics_json: entry.value().metrics_json.clone(),
                })
                .collect(),

            activity: engine
                .activity
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .map(|a| SerializedActivity {
                    id: a.id.clone(),
                    timestamp: a.timestamp,
                    command: a.command.clone(),
                    success: a.success,
                    output: a.output.clone(),
                })
                .collect(),

            notifications: engine
                .notifications
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .map(|n| SerializedNotification {
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
                .collect(),

            skills: engine
                .skills
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .map(|s| SerializedSkill {
                    id: s.id.clone(),
                    name: s.name.clone(),
                    description: s.description.clone(),
                    active: s.active,
                    version: s.version.clone(),
                    category: s.category.clone(),
                    risk_score: s.risk_score,
                    allow_level: s.allow_level.clone(),
                    instructions: s.instructions.clone(),
                    source: s.source.clone(),
                })
                .collect(),

            workflows: engine
                .workflows
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .map(|w| SerializedWorkflow {
                    id: w.id.clone(),
                    name: w.name.clone(),
                    status: w.status.clone(),
                    last_run: w.last_run,
                    duration: w.duration,
                    event_count: w.event_count,
                    source: w.source.clone(),
                })
                .collect(),

            models: engine
                .models
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .map(|m| SerializedModelInfo {
                    id: m.id.clone(),
                    name: m.name.clone(),
                    description: m.description.clone(),
                    path: m.path.clone(),
                    parameter_count: m.parameter_count,
                    state: m.state.clone(),
                    loaded_at: m.loaded_at.clone(),
                    source: m.source.clone(),
                })
                .collect(),

            training_runs: engine
                .training_runs
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .map(|r| SerializedTrainingRun {
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
                    data_points: r
                        .data_points
                        .iter()
                        .map(|p| SerializedTrainingDataPoint {
                            step: p.step,
                            train_loss: p.train_loss,
                            val_loss: p.val_loss,
                            learning_rate: p.learning_rate,
                            accuracy: p.accuracy,
                            grad_norm: p.grad_norm,
                        })
                        .collect(),
                    source: r.source.clone(),
                })
                .collect(),

            engine_mode: if engine.is_demo_mode() {
                "demo".to_string()
            } else {
                "production".to_string()
            },
            config: engine
                .config
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        let json = serde_json::to_string_pretty(&snapshot)
            .map_err(|e| format!("Serialization error: {}", e))?;

        if let Some(parent) = resolved_path.parent() {
            fs::create_dir_all(parent).map_err(|e| format!("Failed to create directory: {}", e))?;
        }

        fs::write(&resolved_path, &json).map_err(|e| format!("Write error: {}", e))?;

        self.set_dirty();
        Ok(())
    }

    pub fn load(&self, engine: &DataEngineInner) -> Result<(), String> {
        let resolved_path = resolve_data_path(&self.save_path)?;

        if !resolved_path.exists() {
            return Err("No save file found".to_string());
        }

        let json = fs::read_to_string(&resolved_path).map_err(|e| format!("Read error: {}", e))?;

        let snapshot: EngineSnapshot =
            serde_json::from_str(&json).map_err(|e| format!("Deserialization error: {}", e))?;

        // Clear and restore agents
        engine.agents.clear();
        for agent in &snapshot.agents {
            engine.agents.insert(
                agent.id.clone(),
                crate::engine::InternalAgent {
                    state: agent.state.clone(),
                    role: agent.role.clone(),
                    metrics_json: agent.metrics_json.clone(),
                },
            );
        }

        // Restore activity
        {
            let mut activity = engine.activity.write().unwrap_or_else(|e| e.into_inner());
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
            let mut notifications = engine
                .notifications
                .write()
                .unwrap_or_else(|e| e.into_inner());
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

        // Restore demo/live catalogs
        {
            let mut skills = engine.skills.write().unwrap_or_else(|e| e.into_inner());
            skills.clear();
            for s in &snapshot.skills {
                skills.push_back(crate::engine::InternalSkill {
                    id: s.id.clone(),
                    name: s.name.clone(),
                    description: s.description.clone(),
                    active: s.active,
                    version: s.version.clone(),
                    category: s.category.clone(),
                    risk_score: s.risk_score,
                    allow_level: s.allow_level.clone(),
                    instructions: s.instructions.clone(),
                    source: s.source.clone(),
                });
            }
        }

        {
            let mut workflows = engine.workflows.write().unwrap_or_else(|e| e.into_inner());
            workflows.clear();
            for w in &snapshot.workflows {
                workflows.push_back(crate::engine::InternalWorkflow {
                    id: w.id.clone(),
                    name: w.name.clone(),
                    status: w.status.clone(),
                    last_run: w.last_run,
                    duration: w.duration,
                    event_count: w.event_count,
                    source: w.source.clone(),
                });
            }
        }

        {
            let mut models = engine.models.write().unwrap_or_else(|e| e.into_inner());
            models.clear();
            for m in &snapshot.models {
                models.push_back(crate::engine::InternalModelInfo {
                    id: m.id.clone(),
                    name: m.name.clone(),
                    description: m.description.clone(),
                    path: m.path.clone(),
                    parameter_count: m.parameter_count,
                    state: m.state.clone(),
                    loaded_at: m.loaded_at.clone(),
                    source: m.source.clone(),
                });
            }
        }

        {
            let mut training_runs = engine
                .training_runs
                .write()
                .unwrap_or_else(|e| e.into_inner());
            training_runs.clear();
            for run in &snapshot.training_runs {
                training_runs.push(crate::engine::InternalTrainingRun {
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
                    data_points: run
                        .data_points
                        .iter()
                        .map(|p| crate::engine::InternalDataPoint {
                            step: p.step,
                            train_loss: p.train_loss,
                            val_loss: p.val_loss,
                            learning_rate: p.learning_rate,
                            accuracy: p.accuracy,
                            grad_norm: p.grad_norm,
                        })
                        .collect(),
                    source: run.source.clone(),
                });
            }
        }

        // Restore config
        {
            let mut config = engine.config.write().unwrap_or_else(|e| e.into_inner());
            config.clear();
            for (key, value) in &snapshot.config {
                config.insert(key.clone(), value.clone());
            }
            config.insert("engine_mode".to_string(), snapshot.engine_mode.clone());
        }

        Ok(())
    }

    pub fn export_json(&self, engine: &DataEngineInner) -> Result<String, String> {
        let snapshot = EngineSnapshot {
            agents: engine
                .agents
                .iter()
                .map(|entry| SerializedAgent {
                    id: entry.key().clone(),
                    state: entry.value().state.clone(),
                    role: entry.value().role.clone(),
                    metrics_json: entry.value().metrics_json.clone(),
                })
                .collect(),

            activity: engine
                .activity
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .map(|a| SerializedActivity {
                    id: a.id.clone(),
                    timestamp: a.timestamp,
                    command: a.command.clone(),
                    success: a.success,
                    output: a.output.clone(),
                })
                .collect(),

            notifications: engine
                .notifications
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .map(|n| SerializedNotification {
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
                .collect(),

            skills: engine
                .skills
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .map(|s| SerializedSkill {
                    id: s.id.clone(),
                    name: s.name.clone(),
                    description: s.description.clone(),
                    active: s.active,
                    version: s.version.clone(),
                    category: s.category.clone(),
                    risk_score: s.risk_score,
                    allow_level: s.allow_level.clone(),
                    instructions: s.instructions.clone(),
                    source: s.source.clone(),
                })
                .collect(),

            workflows: engine
                .workflows
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .map(|w| SerializedWorkflow {
                    id: w.id.clone(),
                    name: w.name.clone(),
                    status: w.status.clone(),
                    last_run: w.last_run,
                    duration: w.duration,
                    event_count: w.event_count,
                    source: w.source.clone(),
                })
                .collect(),

            models: engine
                .models
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .map(|m| SerializedModelInfo {
                    id: m.id.clone(),
                    name: m.name.clone(),
                    description: m.description.clone(),
                    path: m.path.clone(),
                    parameter_count: m.parameter_count,
                    state: m.state.clone(),
                    loaded_at: m.loaded_at.clone(),
                    source: m.source.clone(),
                })
                .collect(),

            training_runs: engine
                .training_runs
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .map(|r| SerializedTrainingRun {
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
                    data_points: r
                        .data_points
                        .iter()
                        .map(|p| SerializedTrainingDataPoint {
                            step: p.step,
                            train_loss: p.train_loss,
                            val_loss: p.val_loss,
                            learning_rate: p.learning_rate,
                            accuracy: p.accuracy,
                            grad_norm: p.grad_norm,
                        })
                        .collect(),
                    source: r.source.clone(),
                })
                .collect(),

            engine_mode: if engine.is_demo_mode() {
                "demo".to_string()
            } else {
                "production".to_string()
            },
            config: engine
                .config
                .read()
                .unwrap_or_else(|e| e.into_inner())
                .clone(),
            timestamp: chrono::Utc::now().to_rfc3339(),
        };

        serde_json::to_string_pretty(&snapshot).map_err(|e| format!("Serialization error: {}", e))
    }

    pub fn import_json(&self, engine: &DataEngineInner, json: &str) -> Result<(), String> {
        if json.len() > MAX_JSON_INPUT_SIZE {
            return Err(format!(
                "JSON input too large: {} bytes (max {})",
                json.len(),
                MAX_JSON_INPUT_SIZE
            ));
        }

        let snapshot: EngineSnapshot =
            serde_json::from_str(json).map_err(|e| format!("Deserialization error: {}", e))?;

        if snapshot.agents.len() > MAX_AGENTS {
            return Err(format!(
                "Too many agents: {} (max {})",
                snapshot.agents.len(),
                MAX_AGENTS
            ));
        }
        if snapshot.activity.len() > MAX_ACTIVITY_IMPORT {
            return Err(format!(
                "Too many activity entries: {} (max {})",
                snapshot.activity.len(),
                MAX_ACTIVITY_IMPORT
            ));
        }
        if snapshot.notifications.len() > MAX_NOTIFICATIONS_IMPORT {
            return Err(format!(
                "Too many notifications: {} (max {})",
                snapshot.notifications.len(),
                MAX_NOTIFICATIONS_IMPORT
            ));
        }
        if snapshot.config.len() > MAX_CONFIG_ENTRIES {
            return Err(format!(
                "Too many config entries: {} (max {})",
                snapshot.config.len(),
                MAX_CONFIG_ENTRIES
            ));
        }
        if snapshot.skills.len() > MAX_SKILLS_IMPORT {
            return Err(format!(
                "Too many skills: {} (max {})",
                snapshot.skills.len(),
                MAX_SKILLS_IMPORT
            ));
        }
        if snapshot.workflows.len() > MAX_WORKFLOWS_IMPORT {
            return Err(format!(
                "Too many workflows: {} (max {})",
                snapshot.workflows.len(),
                MAX_WORKFLOWS_IMPORT
            ));
        }
        if snapshot.models.len() > MAX_MODELS_IMPORT {
            return Err(format!(
                "Too many models: {} (max {})",
                snapshot.models.len(),
                MAX_MODELS_IMPORT
            ));
        }
        if snapshot.training_runs.len() > MAX_TRAINING_RUNS_IMPORT {
            return Err(format!(
                "Too many training runs: {} (max {})",
                snapshot.training_runs.len(),
                MAX_TRAINING_RUNS_IMPORT
            ));
        }

        for agent in &snapshot.agents {
            if agent.id.len() > MAX_FIELD_LENGTH
                || agent.state.len() > MAX_FIELD_LENGTH
                || agent.role.len() > MAX_FIELD_LENGTH
                || agent.metrics_json.len() > MAX_FIELD_LENGTH
            {
                return Err("Agent field exceeds maximum length".to_string());
            }
        }

        for a in &snapshot.activity {
            if a.id.len() > MAX_FIELD_LENGTH
                || a.command.len() > MAX_FIELD_LENGTH
                || a.output.len() > MAX_FIELD_LENGTH
            {
                return Err("Activity field exceeds maximum length".to_string());
            }
        }

        for n in &snapshot.notifications {
            if n.id.len() > MAX_FIELD_LENGTH
                || n.channel.len() > MAX_FIELD_LENGTH
                || n.priority.len() > MAX_FIELD_LENGTH
                || n.category.len() > MAX_FIELD_LENGTH
                || n.title.len() > MAX_FIELD_LENGTH
                || n.body.len() > MAX_FIELD_LENGTH
            {
                return Err("Notification field exceeds maximum length".to_string());
            }
        }

        for s in &snapshot.skills {
            if s.id.len() > MAX_FIELD_LENGTH
                || s.name.len() > MAX_FIELD_LENGTH
                || s.description.len() > MAX_FIELD_LENGTH
                || s.version.len() > MAX_FIELD_LENGTH
                || s.category.len() > MAX_FIELD_LENGTH
                || s.allow_level.len() > MAX_FIELD_LENGTH
                || s.instructions.len() > MAX_FIELD_LENGTH
                || s.source.len() > MAX_FIELD_LENGTH
            {
                return Err("Skill field exceeds maximum length".to_string());
            }
        }

        for w in &snapshot.workflows {
            if w.id.len() > MAX_FIELD_LENGTH
                || w.name.len() > MAX_FIELD_LENGTH
                || w.status.len() > MAX_FIELD_LENGTH
                || w.source.len() > MAX_FIELD_LENGTH
            {
                return Err("Workflow field exceeds maximum length".to_string());
            }
        }

        for m in &snapshot.models {
            if m.id.len() > MAX_FIELD_LENGTH
                || m.name.len() > MAX_FIELD_LENGTH
                || m.description.len() > MAX_FIELD_LENGTH
                || m.path.len() > MAX_FIELD_LENGTH
                || m.state.len() > MAX_FIELD_LENGTH
                || m.source.len() > MAX_FIELD_LENGTH
            {
                return Err("Model field exceeds maximum length".to_string());
            }
            if let Some(loaded_at) = &m.loaded_at {
                if loaded_at.len() > MAX_FIELD_LENGTH {
                    return Err("Model loaded_at exceeds maximum length".to_string());
                }
            }
        }

        let total_training_points: usize = snapshot
            .training_runs
            .iter()
            .map(|run| run.data_points.len())
            .sum();
        if total_training_points > MAX_TRAINING_POINTS_IMPORT {
            return Err(format!(
                "Too many training data points: {} (max {})",
                total_training_points, MAX_TRAINING_POINTS_IMPORT
            ));
        }
        for run in &snapshot.training_runs {
            if run.id.len() > MAX_FIELD_LENGTH
                || run.name.len() > MAX_FIELD_LENGTH
                || run.model_id.len() > MAX_FIELD_LENGTH
                || run.status.len() > MAX_FIELD_LENGTH
                || run.source.len() > MAX_FIELD_LENGTH
            {
                return Err("Training run field exceeds maximum length".to_string());
            }
        }

        for (key, value) in &snapshot.config {
            if key.len() > MAX_FIELD_LENGTH || value.len() > MAX_FIELD_LENGTH {
                return Err("Config field exceeds maximum length".to_string());
            }
        }

        let agents: Vec<_> = snapshot
            .agents
            .iter()
            .map(|agent| {
                (
                    agent.id.clone(),
                    crate::engine::InternalAgent {
                        state: agent.state.clone(),
                        role: agent.role.clone(),
                        metrics_json: agent.metrics_json.clone(),
                    },
                )
            })
            .collect();

        let activity: VecDeque<_> = snapshot
            .activity
            .iter()
            .map(|a| crate::engine::InternalActivity {
                id: a.id.clone(),
                timestamp: a.timestamp,
                command: a.command.clone(),
                success: a.success,
                output: a.output.clone(),
            })
            .collect();

        let notifications: VecDeque<_> = snapshot
            .notifications
            .iter()
            .map(|n| crate::engine::InternalNotification {
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
            .collect();

        let skills: VecDeque<_> = snapshot
            .skills
            .iter()
            .map(|s| crate::engine::InternalSkill {
                id: s.id.clone(),
                name: s.name.clone(),
                description: s.description.clone(),
                active: s.active,
                version: s.version.clone(),
                category: s.category.clone(),
                risk_score: s.risk_score,
                allow_level: s.allow_level.clone(),
                instructions: s.instructions.clone(),
                source: s.source.clone(),
            })
            .collect();

        let workflows: VecDeque<_> = snapshot
            .workflows
            .iter()
            .map(|w| crate::engine::InternalWorkflow {
                id: w.id.clone(),
                name: w.name.clone(),
                status: w.status.clone(),
                last_run: w.last_run,
                duration: w.duration,
                event_count: w.event_count,
                source: w.source.clone(),
            })
            .collect();

        let models: VecDeque<_> = snapshot
            .models
            .iter()
            .map(|m| crate::engine::InternalModelInfo {
                id: m.id.clone(),
                name: m.name.clone(),
                description: m.description.clone(),
                path: m.path.clone(),
                parameter_count: m.parameter_count,
                state: m.state.clone(),
                loaded_at: m.loaded_at.clone(),
                source: m.source.clone(),
            })
            .collect();

        let training_runs: Vec<_> = snapshot
            .training_runs
            .iter()
            .map(|run| crate::engine::InternalTrainingRun {
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
                data_points: run
                    .data_points
                    .iter()
                    .map(|p| crate::engine::InternalDataPoint {
                        step: p.step,
                        train_loss: p.train_loss,
                        val_loss: p.val_loss,
                        learning_rate: p.learning_rate,
                        accuracy: p.accuracy,
                        grad_norm: p.grad_norm,
                    })
                    .collect(),
                source: run.source.clone(),
            })
            .collect();

        engine.agents.clear();
        for (id, agent) in agents {
            engine.agents.insert(id, agent);
        }

        {
            let mut w = engine.activity.write().unwrap_or_else(|e| e.into_inner());
            w.clear();
            *w = activity;
        }

        {
            let mut w = engine
                .notifications
                .write()
                .unwrap_or_else(|e| e.into_inner());
            w.clear();
            *w = notifications;
        }

        {
            let mut w = engine.skills.write().unwrap_or_else(|e| e.into_inner());
            w.clear();
            *w = skills;
        }

        {
            let mut w = engine.workflows.write().unwrap_or_else(|e| e.into_inner());
            w.clear();
            *w = workflows;
        }

        {
            let mut w = engine.models.write().unwrap_or_else(|e| e.into_inner());
            w.clear();
            *w = models;
        }

        {
            let mut w = engine
                .training_runs
                .write()
                .unwrap_or_else(|e| e.into_inner());
            w.clear();
            *w = training_runs;
        }

        {
            let mut w = engine.config.write().unwrap_or_else(|e| e.into_inner());
            w.clear();
            for (key, value) in &snapshot.config {
                w.insert(key.clone(), value.clone());
            }
            w.insert("engine_mode".to_string(), snapshot.engine_mode.clone());
        }

        Ok(())
    }
}
