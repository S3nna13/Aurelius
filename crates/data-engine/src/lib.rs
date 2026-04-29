mod engine;
mod persistence;

use engine::DataEngineInner;
use napi_derive::napi;

#[napi(object)]
pub struct AgentState {
    pub id: String,
    pub state: String,
    pub role: String,
    pub metrics_json: String,
}

#[napi(object)]
pub struct ActivityEntry {
    pub id: String,
    pub timestamp: f64,
    pub command: String,
    pub success: bool,
    pub output: String,
}

#[napi(object)]
pub struct Notification {
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

#[napi(object)]
pub struct NotificationStats {
    pub unread: u32,
    pub total: u32,
}

#[napi(object)]
pub struct MemoryLayer {
    pub name: String,
    pub entries: u32,
}

#[napi(object)]
pub struct MemoryEntry {
    pub id: String,
    pub content: String,
    pub layer: String,
    pub timestamp: String,
    pub access_count: u32,
    pub importance_score: f64,
}

#[napi(object)]
pub struct LogRecord {
    pub timestamp: String,
    pub level: String,
    pub logger: String,
    pub message: String,
}

#[napi(object)]
pub struct SkillEntry {
    pub id: String,
    pub name: String,
    pub description: String,
    pub active: bool,
    pub version: String,
    pub category: String,
    pub risk_score: f64,
    pub allow_level: String,
    pub source: String,
}

#[napi(object)]
pub struct WorkflowEntry {
    pub id: String,
    pub name: String,
    pub status: String,
    pub last_run: f64,
    pub duration: f64,
    pub event_count: u32,
    pub source: String,
}

#[napi(object)]
pub struct ModelInfo {
    pub id: String,
    pub name: String,
    pub description: String,
    pub path: String,
    pub parameter_count: i64,
    pub state: String,
    pub loaded_at: Option<String>,
    pub source: String,
}

#[napi(object)]
pub struct TrainingDataPoint {
    pub step: u32,
    pub train_loss: f64,
    pub val_loss: f64,
    pub learning_rate: f64,
    pub accuracy: f64,
    pub grad_norm: f64,
}

#[napi(object)]
pub struct TrainingRunSummary {
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
    pub data_point_count: u32,
    pub source: String,
}

#[napi(object)]
pub struct TrainingRunDetail {
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
    pub steps: Vec<u32>,
    pub train_losses: Vec<f64>,
    pub val_losses: Vec<f64>,
    pub learning_rates: Vec<f64>,
    pub accuracies: Vec<f64>,
    pub source: String,
}

#[napi(object)]
pub struct SearchResult {
    pub matches: u32,
    pub entries: Vec<LogRecord>,
}

#[napi(object)]
pub struct SystemStats {
    pub agent_count: u32,
    pub activity_count: u32,
    pub notification_count: u32,
    pub notification_unread: u32,
    pub memory_entry_count: u32,
    pub log_count: u32,
}

#[napi]
#[allow(dead_code)]
pub struct DataEngine {
    inner: DataEngineInner,
    persistence: std::sync::Mutex<Option<persistence::Persistence>>,
}

#[napi]
impl DataEngine {
    #[napi(constructor)]
    pub fn new() -> Self {
        DataEngine {
            inner: DataEngineInner::new(),
            persistence: std::sync::Mutex::new(None),
        }
    }

    #[napi]
    pub fn get_agent(&self, id: String) -> Option<AgentState> {
        self.inner.get_agent(&id)
    }

    #[napi]
    pub fn list_agents(&self) -> Vec<AgentState> {
        self.inner.list_agents()
    }

    #[napi]
    pub fn upsert_agent(&self, id: String, state: String, role: String, metrics_json: String) {
        self.inner.upsert_agent(&id, &state, &role, &metrics_json)
    }

    #[napi]
    pub fn delete_agent(&self, id: String) -> bool {
        self.inner.delete_agent(&id)
    }

    #[napi]
    pub fn append_activity(&self, command: String, success: bool, output: String) -> ActivityEntry {
        self.inner.append_activity(&command, success, &output)
    }

    #[napi]
    pub fn get_activity(&self, limit: Option<u32>) -> Vec<ActivityEntry> {
        self.inner.get_activity(limit)
    }

    #[napi]
    pub fn search_activity(&self, query: String, limit: Option<u32>) -> Vec<ActivityEntry> {
        self.inner.search_activity(&query, limit)
    }

    #[napi]
    pub fn add_notification(&self, channel: String, priority: String, category: String, title: String, body: String) -> Notification {
        self.inner.add_notification(&channel, &priority, &category, &title, &body)
    }

    #[napi]
    pub fn get_notifications(&self, category: Option<String>, priority: Option<String>, read: Option<bool>, limit: Option<u32>) -> Vec<Notification> {
        self.inner.get_notifications(category.as_deref(), priority.as_deref(), read, limit)
    }

    #[napi]
    pub fn mark_notification_read(&self, id: String) -> bool {
        self.inner.mark_notification_read(&id)
    }

    #[napi]
    pub fn mark_all_notifications_read(&self, category: Option<String>) -> u32 {
        self.inner.mark_all_notifications_read(category.as_deref())
    }

    #[napi]
    pub fn get_notification_stats(&self) -> NotificationStats {
        self.inner.get_notification_stats()
    }

    #[napi]
    pub fn get_memory_layers(&self) -> Vec<MemoryLayer> {
        self.inner.get_memory_layers()
    }

    #[napi]
    pub fn add_memory_entry(&self, layer: String, content: String) {
        self.inner.add_memory_entry(&layer, &content)
    }

    #[napi]
    pub fn get_memory_entries(&self, layer: Option<String>, query: Option<String>, limit: Option<u32>) -> Vec<MemoryEntry> {
        self.inner.get_memory_entries(layer.as_deref(), query.as_deref(), limit)
    }

    #[napi]
    pub fn delete_memory_entry(&self, id: String) -> bool {
        self.inner.delete_memory_entry(&id)
    }

    #[napi]
    pub fn get_config(&self, key: String) -> Option<String> {
        self.inner.get_config(&key)
    }

    #[napi]
    pub fn get_all_config(&self) -> std::collections::HashMap<String, String> {
        self.inner.get_all_config()
    }

    #[napi]
    pub fn set_config(&self, key: String, value: String) {
        self.inner.set_config(&key, &value)
    }

    #[napi]
    pub fn append_log(&self, level: String, logger: String, message: String) {
        self.inner.append_log(&level, &logger, &message)
    }

    #[napi]
    pub fn get_logs(&self, level: Option<String>, query: Option<String>, limit: Option<u32>) -> Vec<LogRecord> {
        self.inner.get_logs(level.as_deref(), query.as_deref(), limit)
    }

    #[napi]
    pub fn search_logs(&self, query: String, level: Option<String>, limit: Option<u32>) -> Vec<LogRecord> {
        self.inner.search_logs(&query, level.as_deref(), limit)
    }

    #[napi]
    pub fn get_stats(&self) -> SystemStats {
        self.inner.get_stats()
    }

    #[napi]
    pub fn clear_activity(&self) -> u32 {
        self.inner.clear_activity()
    }

    #[napi]
    pub fn clear_notifications(&self) -> u32 {
        self.inner.clear_notifications()
    }

    #[napi]
    pub fn clear_logs(&self) -> u32 {
        self.inner.clear_logs()
    }

    #[napi]
    pub fn export_json(&self) -> String {
        use persistence::Persistence;
        let p = Persistence::new("", 0);
        p.export_json(&self.inner).unwrap_or_else(|e| format!("{{\"error\": \"{e}\"}}"))
    }

    #[napi]
    pub fn import_json(&self, json: String) -> bool {
        use persistence::Persistence;
        let p = Persistence::new("", 0);
        p.import_json(&self.inner, &json).is_ok()
    }

    #[napi]
    pub fn save_to_file(&self, path: String) -> bool {
        use persistence::Persistence;
        if persistence::resolve_data_path(&path).is_err() {
            return false;
        }
        let p = Persistence::new(&path, 0);
        p.save(&self.inner).is_ok()
    }

    #[napi]
    pub fn load_from_file(&self, path: String) -> bool {
        use persistence::Persistence;
        if persistence::resolve_data_path(&path).is_err() {
            return false;
        }
        let p = Persistence::new(&path, 0);
        p.load(&self.inner).is_ok()
    }

    // -- Skills --

    #[napi]
    pub fn list_skills(&self) -> Vec<SkillEntry> {
        self.inner.list_skills()
    }

    #[napi]
    pub fn get_skill(&self, id: String) -> Option<SkillEntry> {
        self.inner.get_skill(&id)
    }

    // -- Workflows --

    #[napi]
    pub fn list_workflows(&self) -> Vec<WorkflowEntry> {
        self.inner.list_workflows()
    }

    #[napi]
    pub fn get_workflow(&self, id: String) -> Option<WorkflowEntry> {
        self.inner.get_workflow(&id)
    }

    #[napi]
    pub fn update_workflow_status(&self, id: String, status: String) -> bool {
        self.inner.update_workflow_status(&id, &status)
    }

    // -- Models --

    #[napi]
    pub fn list_models(&self) -> Vec<ModelInfo> {
        self.inner.list_models()
    }

    #[napi]
    pub fn get_model(&self, id: String) -> Option<ModelInfo> {
        self.inner.get_model(&id)
    }

    #[napi]
    pub fn set_model_state(&self, id: String, state: String) -> bool {
        self.inner.set_model_state(&id, &state)
    }

    // -- Training Runs --

    #[napi]
    pub fn list_training_runs(&self) -> Vec<TrainingRunSummary> {
        self.inner.list_training_runs()
    }

    #[napi]
    pub fn get_training_run(&self, id: String) -> Option<TrainingRunDetail> {
        self.inner.get_training_run(&id)
    }

    #[napi]
    pub fn create_training_run(&self, name: String, model_id: String, total_epochs: u32) -> TrainingRunSummary {
        self.inner.create_training_run(&name, &model_id, total_epochs)
    }
}
