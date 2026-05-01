use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::{Duration, Instant};

use tokio::sync::Mutex;

#[derive(Clone)]
pub struct RateLimiter {
    inner: Arc<Mutex<RateLimiterInner>>,
    rps: u64,
    burst: u64,
}

struct Bucket {
    tokens: f64,
    last_refill: Instant,
}

struct RateLimiterInner {
    buckets: HashMap<String, Bucket>,
}

impl RateLimiter {
    pub fn new(rps: u64, burst: u64) -> Self {
        let inner = Arc::new(Mutex::new(RateLimiterInner {
            buckets: HashMap::new(),
        }));

        let inner_clone = inner.clone();
        tokio::spawn(async move {
            loop {
                tokio::time::sleep(Duration::from_secs(60)).await;
                let mut inner = inner_clone.lock().await;
                let cutoff = Instant::now() - Duration::from_secs(120);
                inner.buckets.retain(|_, b| b.last_refill > cutoff);
            }
        });

        RateLimiter { inner, rps, burst }
    }

    pub async fn check(&self, key: &str) -> RateLimitResult {
        let mut inner = self.inner.lock().await;
        let now = Instant::now();
        let bucket = inner.buckets.entry(key.to_string()).or_insert(Bucket {
            tokens: self.burst as f64,
            last_refill: now,
        });

        let elapsed = now.duration_since(bucket.last_refill).as_secs_f64();
        let refill = elapsed * self.rps as f64;
        bucket.tokens = (bucket.tokens + refill).min(self.burst as f64);
        bucket.last_refill = now;

        if bucket.tokens >= 1.0 {
            bucket.tokens -= 1.0;
            RateLimitResult::Allowed {
                remaining: bucket.tokens as u64,
                reset: (self.burst as f64 - bucket.tokens) as u64,
            }
        } else {
            RateLimitResult::Denied {
                retry_after: (1.0 / self.rps as f64).ceil() as u64,
            }
        }
    }

    pub async fn check_socket(&self, addr: &SocketAddr) -> RateLimitResult {
        self.check(&addr.ip().to_string()).await
    }
}

#[derive(Debug)]
pub enum RateLimitResult {
    Allowed { remaining: u64, reset: u64 },
    Denied { retry_after: u64 },
}
