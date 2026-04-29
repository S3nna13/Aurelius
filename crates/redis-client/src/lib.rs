use std::sync::Arc;
use std::time::Duration;

use dashmap::DashMap;
use napi::bindgen_prelude::*;
use napi_derive::napi;
use redis::aio::ConnectionManager;
use redis::AsyncCommands;
use tokio::runtime::Runtime;
use tokio::sync::Mutex;

#[napi(object)]
pub struct RedisConfig {
    pub url: String,
    pub pool_size: Option<u32>,
    pub timeout_ms: Option<u32>,
    pub retry_count: Option<u32>,
}

#[napi(object)]
pub struct RedisInfo {
    pub connected: bool,
    pub latency_ms: f64,
    pub keys_count: Option<i64>,
    pub server_version: Option<String>,
    pub memory_used: Option<i64>,
}

#[napi(object)]
pub struct RedisHashEntry {
    pub field: String,
    pub value: String,
}

type RedisPool = Arc<Mutex<Option<ConnectionManager>>>;

#[napi]
pub struct RedisClient {
    pool: RedisPool,
    config: RedisConfig,
    runtime: Arc<Runtime>,
    connected: Arc<std::sync::atomic::AtomicBool>,
}

#[napi]
impl RedisClient {
    #[napi(constructor)]
    pub fn new(config: RedisConfig) -> Self {
        let runtime = Arc::new(Runtime::new().unwrap());
        let pool = Arc::new(Mutex::new(None));

        RedisClient {
            pool,
            config,
            runtime,
            connected: Arc::new(std::sync::atomic::AtomicBool::new(false)),
        }
    }

    async fn connect(&self) -> Result<()> {
        let url = self.config.url.clone();
        let timeout = Duration::from_millis(self.config.timeout_ms.unwrap_or(5000) as u64);

        let client = redis::Client::open(url.as_str())
            .map_err(|e| napi::Error::from_reason(format!("Redis connect error: {}", e)))?;

        let conn = tokio::time::timeout(
            timeout,
            ConnectionManager::new(client),
        )
            .await
            .map_err(|_| napi::Error::from_reason("Redis connection timeout"))?
            .map_err(|e| napi::Error::from_reason(format!("Redis connection error: {}", e)))?;

        let mut pool = self.pool.lock().await;
        *pool = Some(conn);
        self.connected.store(true, std::sync::atomic::Ordering::Relaxed);
        Ok(())
    }

    async fn get_conn(&self) -> Result<()> {
        let pool = self.pool.lock().await;
        if pool.is_none() {
            drop(pool);
            self.connect().await?;
        }
        Ok(())
    }

    #[napi]
    pub async fn ping(&self) -> Result<String> {
        self.connect().await?;
        let pool = self.pool.clone();
        let mut guard = pool.lock().await;
        let conn = guard.as_mut().ok_or_else(|| napi::Error::from_reason("Not connected"))?;

        let result: String = conn
            .ping()
            .await
            .map_err(|e| napi::Error::from_reason(format!("Redis ping error: {}", e)))?;

        Ok(result)
    }

    #[napi]
    pub async fn set(&self, key: String, value: String, ttl_seconds: Option<i64>) -> Result<()> {
        self.connect().await?;
        let pool = self.pool.clone();
        let mut guard = pool.lock().await;
        let conn = guard.as_mut().ok_or_else(|| napi::Error::from_reason("Not connected"))?;

        if let Some(ttl) = ttl_seconds {
            let _: () = conn.set_ex(&key, &value, ttl as u64)
                .await
                .map_err(|e| napi::Error::from_reason(format!("Redis set error: {}", e)))?;
        } else {
            let _: () = conn.set(&key, &value)
                .await
                .map_err(|e| napi::Error::from_reason(format!("Redis set error: {}", e)))?;
        }

        Ok(())
    }

    #[napi]
    pub async fn get(&self, key: String) -> Result<Option<String>> {
        self.connect().await?;
        let pool = self.pool.clone();
        let mut guard = pool.lock().await;
        let conn = guard.as_mut().ok_or_else(|| napi::Error::from_reason("Not connected"))?;

        let result: Option<String> = conn
            .get(key)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Redis get error: {}", e)))?;

        Ok(result)
    }

    #[napi]
    pub async fn del(&self, key: String) -> Result<bool> {
        self.connect().await?;
        let pool = self.pool.clone();
        let mut guard = pool.lock().await;
        let conn = guard.as_mut().ok_or_else(|| napi::Error::from_reason("Not connected"))?;

        let result: u32 = conn
            .del(key)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Redis del error: {}", e)))?;

        Ok(result > 0)
    }

    #[napi]
    pub async fn exists(&self, key: String) -> Result<bool> {
        self.connect().await?;
        let pool = self.pool.clone();
        let mut guard = pool.lock().await;
        let conn = guard.as_mut().ok_or_else(|| napi::Error::from_reason("Not connected"))?;

        let result: bool = conn
            .exists(key)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Redis exists error: {}", e)))?;

        Ok(result)
    }

    #[napi]
    pub async fn expire(&self, key: String, seconds: i64) -> Result<bool> {
        self.connect().await?;
        let pool = self.pool.clone();
        let mut guard = pool.lock().await;
        let conn = guard.as_mut().ok_or_else(|| napi::Error::from_reason("Not connected"))?;

        let result: bool = conn
            .expire(&key, seconds as i64)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Redis expire error: {}", e)))?;

        Ok(result)
    }

    #[napi]
    pub async fn ttl(&self, key: String) -> Result<i64> {
        self.connect().await?;
        let pool = self.pool.clone();
        let mut guard = pool.lock().await;
        let conn = guard.as_mut().ok_or_else(|| napi::Error::from_reason("Not connected"))?;

        let result: i64 = conn
            .ttl(key)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Redis ttl error: {}", e)))?;

        Ok(result)
    }

    #[napi]
    pub async fn incr(&self, key: String) -> Result<i64> {
        self.connect().await?;
        let pool = self.pool.clone();
        let mut guard = pool.lock().await;
        let conn = guard.as_mut().ok_or_else(|| napi::Error::from_reason("Not connected"))?;

        let result: i64 = conn
            .incr(key, 1)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Redis incr error: {}", e)))?;

        Ok(result)
    }

    #[napi]
    pub async fn hset(&self, key: String, field: String, value: String) -> Result<bool> {
        self.connect().await?;
        let pool = self.pool.clone();
        let mut guard = pool.lock().await;
        let conn = guard.as_mut().ok_or_else(|| napi::Error::from_reason("Not connected"))?;

        let result: bool = conn
            .hset(key, field, value)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Redis hset error: {}", e)))?;

        Ok(result)
    }

    #[napi]
    pub async fn hget(&self, key: String, field: String) -> Result<Option<String>> {
        self.connect().await?;
        let pool = self.pool.clone();
        let mut guard = pool.lock().await;
        let conn = guard.as_mut().ok_or_else(|| napi::Error::from_reason("Not connected"))?;

        let result: Option<String> = conn
            .hget(key, field)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Redis hget error: {}", e)))?;

        Ok(result)
    }

    #[napi]
    pub async fn hgetall(&self, key: String) -> Result<Vec<RedisHashEntry>> {
        self.connect().await?;
        let pool = self.pool.clone();
        let mut guard = pool.lock().await;
        let conn = guard.as_mut().ok_or_else(|| napi::Error::from_reason("Not connected"))?;

        let result: Vec<(String, String)> = conn
            .hgetall(key)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Redis hgetall error: {}", e)))?;

        Ok(result.into_iter().map(|(f, v)| RedisHashEntry { field: f, value: v }).collect())
    }

    #[napi]
    pub async fn lpush(&self, key: String, value: String) -> Result<u32> {
        self.connect().await?;
        let pool = self.pool.clone();
        let mut guard = pool.lock().await;
        let conn = guard.as_mut().ok_or_else(|| napi::Error::from_reason("Not connected"))?;

        let result: u32 = conn
            .lpush(key, value)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Redis lpush error: {}", e)))?;

        Ok(result)
    }

    #[napi]
    pub async fn rpop(&self, key: String) -> Result<Option<String>> {
        self.connect().await?;
        let pool = self.pool.clone();
        let mut guard = pool.lock().await;
        let conn = guard.as_mut().ok_or_else(|| napi::Error::from_reason("Not connected"))?;

        let result: Option<String> = conn
            .rpop(key, None)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Redis rpop error: {}", e)))?;

        Ok(result)
    }

    #[napi]
    pub async fn lrange(&self, key: String, start: i64, stop: i64) -> Result<Vec<String>> {
        self.connect().await?;
        let pool = self.pool.clone();
        let mut guard = pool.lock().await;
        let conn = guard.as_mut().ok_or_else(|| napi::Error::from_reason("Not connected"))?;

        let result: Vec<String> = conn
            .lrange(&key, start as isize, stop as isize)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Redis lrange error: {}", e)))?;

        Ok(result)
    }

    #[napi]
    pub async fn publish(&self, channel: String, message: String) -> Result<u32> {
        self.connect().await?;
        let pool = self.pool.clone();
        let mut guard = pool.lock().await;
        let conn = guard.as_mut().ok_or_else(|| napi::Error::from_reason("Not connected"))?;

        let result: u32 = conn
            .publish(channel, message)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Redis publish error: {}", e)))?;

        Ok(result)
    }

    #[napi]
    pub async fn info(&self) -> Result<RedisInfo> {
        self.connect().await?;
        let pool = self.pool.clone();
        let mut guard = pool.lock().await;
        let conn = guard.as_mut().ok_or_else(|| napi::Error::from_reason("Not connected"))?;

        let start = std::time::Instant::now();

        let ping_result: String = conn
            .ping()
            .await
            .map_err(|e| napi::Error::from_reason(format!("Redis ping error: {}", e)))?;

        let latency = start.elapsed().as_secs_f64() * 1000.0;

        let info_str: String = redis::cmd("INFO")
            .query_async(&mut *conn)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Redis info error: {}", e)))?;

        let keys: u64 = redis::cmd("DBSIZE")
            .query_async(&mut *conn)
            .await
            .map_err(|e| napi::Error::from_reason(format!("Redis dbsize error: {}", e)))?;

        // Parse basic info from INFO output
        let mut server_version = None;
        let mut memory_used = None;
        for line in info_str.lines() {
            if let Some(val) = line.strip_prefix("redis_version:") {
                server_version = Some(val.trim().to_string());
            }
            if let Some(val) = line.strip_prefix("used_memory:") {
                memory_used = val.trim().parse::<u64>().ok().map(|m| m as i64);
            }
        }

        Ok(RedisInfo {
            connected: ping_result.to_uppercase() == "PONG",
            latency_ms: (latency * 100.0).round() / 100.0,
            keys_count: Some(keys as i64),
            server_version,
            memory_used,
        })
    }

    #[napi]
    pub fn is_connected(&self) -> bool {
        self.connected.load(std::sync::atomic::Ordering::Relaxed)
    }
}
