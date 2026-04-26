use chrono::{DateTime, Utc};
use pyo3::prelude::*;
use pyo3::types::PyDict;
use redb::{Database, ReadableTable, TableDefinition};
use serde::{Deserialize, Serialize};
use std::path::Path;
use std::sync::Arc;
use ulid::Ulid;

const EVENTS_TABLE: TableDefinition<&str, &str> = TableDefinition::new("events");
const USER_INDEX_TABLE: TableDefinition<&str, &str> = TableDefinition::new("user_index");

#[derive(Serialize, Deserialize, Debug, Clone)]
struct Event {
    id: String,
    user_id: String,
    event_type: String,
    payload: serde_json::Value,
    timestamp: DateTime<Utc>,
}

pub struct UsageStoreInner {
    db: Arc<Database>,
}

impl UsageStoreInner {
    fn open<P: AsRef<Path>>(path: P) -> Result<Self, redb::Error> {
        let db = Database::create(path)?;
        let txn = db.begin_write()?;
        {
            let _ = txn.open_table(EVENTS_TABLE)?;
            let _ = txn.open_table(USER_INDEX_TABLE)?;
        }
        txn.commit()?;
        Ok(Self { db: Arc::new(db) })
    }

    fn log_event(
        &self,
        user_id: &str,
        event_type: &str,
        payload: serde_json::Value,
    ) -> Result<String, redb::Error> {
        let id = Ulid::new().to_string();
        let event = Event {
            id: id.clone(),
            user_id: user_id.to_string(),
            event_type: event_type.to_string(),
            payload,
            timestamp: Utc::now(),
        };
        let json = serde_json::to_string(&event).unwrap();

        let txn = self.db.begin_write()?;
        {
            let mut events = txn.open_table(EVENTS_TABLE)?;
            events.insert(id.as_str(), json.as_str())?;

            let mut index = txn.open_table(USER_INDEX_TABLE)?;
            let existing = index.get(user_id)?.map(|g| g.value().to_string());
            let updated = match existing {
                Some(s) if !s.is_empty() => format!("{},{}" , s, id),
                _ => id.clone(),
            };
            index.insert(user_id, updated.as_str())?;
        }
        txn.commit()?;
        Ok(id)
    }

    fn query_events(
        &self,
        user_id: &str,
        since: Option<DateTime<Utc>>,
        event_type: Option<&str>,
        limit: usize,
    ) -> Result<Vec<Event>, redb::Error> {
        let txn = self.db.begin_read()?;
        let index = txn.open_table(USER_INDEX_TABLE)?;
        let ids = index
            .get(user_id)?
            .map(|g| g.value().to_string())
            .unwrap_or_default();

        if ids.is_empty() {
            return Ok(vec![]);
        }

        let events_table = txn.open_table(EVENTS_TABLE)?;
        let mut out = Vec::new();
        for id in ids.split(',') {
            if let Some(guard) = events_table.get(id)? {
                let ev: Event = serde_json::from_str(guard.value()).unwrap();
                if let Some(ref t) = since {
                    if ev.timestamp < *t {
                        continue;
                    }
                }
                if let Some(et) = event_type {
                    if ev.event_type != et {
                        continue;
                    }
                }
                out.push(ev);
                if out.len() >= limit {
                    break;
                }
            }
        }
        Ok(out)
    }

    fn event_count(&self, user_id: &str) -> Result<usize, redb::Error> {
        let txn = self.db.begin_read()?;
        let index = txn.open_table(USER_INDEX_TABLE)?;
        let ids = index
            .get(user_id)?
            .map(|g| g.value().to_string())
            .unwrap_or_default();
        Ok(ids.split(',').filter(|s| !s.is_empty()).count())
    }
}

#[pyclass]
pub struct UsageStore {
    inner: UsageStoreInner,
}

#[pymethods]
impl UsageStore {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let inner = UsageStoreInner::open(path)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;
        Ok(Self { inner })
    }

    fn log_event(
        &self,
        user_id: &str,
        event_type: &str,
        payload: &Bound<'_, PyDict>,
    ) -> PyResult<String> {
        let payload: serde_json::Value = python_dict_to_json(payload)?;
        self.inner
            .log_event(user_id, event_type, payload)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
    }

    #[pyo3(signature = (user_id, since=None, event_type=None, limit=100))]
    fn query_events(
        &self,
        user_id: &str,
        since: Option<&str>,
        event_type: Option<&str>,
        limit: usize,
    ) -> PyResult<Vec<PyObject>> {
        let since = match since {
            Some(s) => Some(
                DateTime::parse_from_rfc3339(s)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{e}")))?
                    .with_timezone(&Utc),
            ),
            None => None,
        };
        let events = self
            .inner
            .query_events(user_id, since, event_type, limit)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))?;

        Python::with_gil(|py| {
            events
                .into_iter()
                .map(|ev| event_to_python(py, ev))
                .collect::<PyResult<Vec<_>>>()
        })
    }

    fn event_count(&self, user_id: &str) -> PyResult<usize> {
        self.inner
            .event_count(user_id)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("{e}")))
    }
}

fn python_dict_to_json(dict: &Bound<'_, PyDict>) -> PyResult<serde_json::Value> {
    let json_str = Python::with_gil(|py| {
        let json_module = py.import("json")?;
        json_module
            .getattr("dumps")?
            .call1((dict,))?
            .extract::<String>()
    })?;
    Ok(serde_json::from_str(&json_str).unwrap())
}

fn event_to_python(py: Python, ev: Event) -> PyResult<PyObject> {
    let dict = PyDict::new(py);
    dict.set_item("id", ev.id)?;
    dict.set_item("user_id", ev.user_id)?;
    dict.set_item("event_type", ev.event_type)?;
    let json_module = py.import("json")?;
    let payload_py = json_module
        .getattr("loads")?
        .call1((ev.payload.to_string(),))?;
    dict.set_item("payload", payload_py)?;
    dict.set_item("timestamp", ev.timestamp.to_rfc3339())?;
    Ok(dict.into_any().unbind())
}

#[pymodule]
fn aurelius_usage_store(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<UsageStore>()?;
    Ok(())
}
