use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use dashmap::DashMap;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct Metrics {
    inner: Arc<MetricsInner>,
}

struct MetricsInner {
    request_count: DashMap<String, u64>,
    status_counts: DashMap<u16, u64>,
    latencies: DashMap<String, Vec<f64>>,
    total_requests: Arc<std::sync::atomic::AtomicU64>,
    start_time: Instant,
}

impl Metrics {
    pub fn new() -> Self {
        Metrics {
            inner: Arc::new(MetricsInner {
                request_count: DashMap::new(),
                status_counts: DashMap::new(),
                latencies: DashMap::new(),
                total_requests: Arc::new(std::sync::atomic::AtomicU64::new(0)),
                start_time: Instant::now(),
            }),
        }
    }

    pub fn record_request(&self, path: &str, status: u16, latency_ms: f64) {
        self.inner.total_requests.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

        *self.inner.request_count.entry(path.to_string()).or_insert(0) += 1;
        *self.inner.status_counts.entry(status).or_insert(0) += 1;

        let mut latencies = self.inner.latencies.entry("all".to_string()).or_insert_with(Vec::new);
        latencies.push(latency_ms);
        if latencies.len() > 1000 {
            latencies.remove(0);
        }
    }

    pub fn snapshot(&self) -> MetricsSnapshot {
        let uptime = self.inner.start_time.elapsed().as_secs_f64();
        let total = self.inner.total_requests.load(std::sync::atomic::Ordering::Relaxed);

        let requests: HashMap<String, u64> = self.inner.request_count.iter()
            .map(|e| (e.key().clone(), *e.value()))
            .collect();

        let statuses: HashMap<u16, u64> = self.inner.status_counts.iter()
            .map(|e| (*e.key(), *e.value()))
            .collect();

        let latencies = self.inner.latencies.get("all")
            .map(|v| v.clone())
            .unwrap_or_default();

        let p50 = percentile(&latencies, 50.0);
        let p95 = percentile(&latencies, 95.0);
        let p99 = percentile(&latencies, 99.0);

        MetricsSnapshot {
            uptime_secs: uptime,
            total_requests: total,
            requests_per_path: requests,
            status_counts: statuses,
            latency_p50_ms: p50,
            latency_p95_ms: p95,
            latency_p99_ms: p99,
        }
    }
}

fn percentile(data: &[f64], p: f64) -> f64 {
    if data.is_empty() { return 0.0; }
    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let idx = ((p / 100.0) * sorted.len() as f64).ceil() as usize;
    sorted.get(idx.saturating_sub(1)).copied().unwrap_or(0.0)
}

#[derive(Debug, serde::Serialize)]
pub struct MetricsSnapshot {
    pub uptime_secs: f64,
    pub total_requests: u64,
    pub requests_per_path: HashMap<String, u64>,
    pub status_counts: HashMap<u16, u64>,
    pub latency_p50_ms: f64,
    pub latency_p95_ms: f64,
    pub latency_p99_ms: f64,
}
