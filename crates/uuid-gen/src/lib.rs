use napi_derive::napi;
use uuid::Uuid;

#[napi]
pub fn uuid_v4() -> String {
    Uuid::new_v4().to_string()
}

#[napi]
pub fn uuid_v7() -> String {
    Uuid::now_v7().to_string()
}

#[napi]
pub fn uuid_v4_batch(count: u32) -> Vec<String> {
    (0..count).map(|_| Uuid::new_v4().to_string()).collect()
}

#[napi]
pub fn uuid_v7_batch(count: u32) -> Vec<String> {
    (0..count).map(|_| Uuid::now_v7().to_string()).collect()
}

#[napi]
pub fn uuid_nil() -> String {
    Uuid::nil().to_string()
}

#[napi]
pub fn uuid_is_valid(s: String) -> bool {
    Uuid::parse_str(&s).is_ok()
}

#[napi]
pub fn uuid_timestamp(uuid_str: String) -> Option<f64> {
    let uuid = Uuid::parse_str(&uuid_str).ok()?;
    let ts = uuid.get_timestamp()?;
    let unix = ts.to_unix();
    Some(unix as f64 / 1000.0)
}

#[napi]
pub fn uuid_version(uuid_str: String) -> Option<u32> {
    let uuid = Uuid::parse_str(&uuid_str).ok()?;
    Some(uuid.get_version_num() as u32)
}

#[napi]
pub fn uuid_variant(uuid_str: String) -> String {
    Uuid::parse_str(&uuid_str)
        .map(|u| format!("{:?}", u.get_variant()))
        .unwrap_or_default()
}

#[napi]
pub fn uuid_short() -> String {
    Uuid::new_v4().to_string().split('-').next().unwrap().to_string()
}

#[napi]
pub fn uuid_sort_key() -> u128 {
    Uuid::now_v7().as_u128()
}
