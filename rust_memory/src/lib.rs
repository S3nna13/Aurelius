use pyo3::prelude::*;
use std::fs::{File, OpenOptions};
use std::io::{Write, Seek, SeekFrom};

type PageId = u64;

#[derive(Clone, Debug, PartialEq)]
enum MemoryLocation {
    Gpu,
    Cpu,
}

#[derive(Clone, Debug)]
struct MemoryPage {
    id: PageId,
    access_count: u64,
    last_access: u64,
    priority: f32,
    size_bytes: u64,
    location: MemoryLocation,
}

#[pyclass]
#[derive(Debug)]
pub struct MemoryPageTable {
    pages: Vec<MemoryPage>,
    capacity: usize,
    clock: u64,
    total_gpu_bytes: u64,
    gpu_budget: u64,
}

#[pymethods]
impl MemoryPageTable {
    #[new]
    pub fn new(capacity: usize, gpu_budget_mb: u64) -> Self {
        MemoryPageTable {
            pages: Vec::with_capacity(capacity),
            capacity,
            clock: 0,
            total_gpu_bytes: 0,
            gpu_budget: gpu_budget_mb * 1024 * 1024,
        }
    }

    pub fn register_page(&mut self, id: u64, priority: f32, size_bytes: u64, on_gpu: bool) -> String {
        self.clock += 1;
        if self.pages.len() >= self.capacity {
            return format!("full:{}", self.capacity);
        }
        if self.pages.iter().any(|p| p.id == id) {
            return format!("exists:{}", id);
        }
        let location = if on_gpu { MemoryLocation::Gpu } else { MemoryLocation::Cpu };
        if on_gpu {
            self.total_gpu_bytes = self.total_gpu_bytes.saturating_add(size_bytes);
        }
        self.pages.push(MemoryPage {
            id, access_count: 0, last_access: self.clock,
            priority, size_bytes, location,
        });
        "ok".to_string()
    }

    pub fn access(&mut self, id: u64) -> String {
        self.clock += 1;
        for page in &mut self.pages {
            if page.id == id {
                page.access_count += 1;
                page.last_access = self.clock;
                return match page.location {
                    MemoryLocation::Gpu => "gpu",
                    MemoryLocation::Cpu => "cpu",
                }.to_string();
            }
        }
        "absent".to_string()
    }

    pub fn promote_to_gpu(&mut self, id: u64) -> String {
        self.clock += 1;
        let idx = self.pages.iter().position(|p| p.id == id);
        match idx {
            None => "absent".to_string(),
            Some(i) => {
                if self.pages[i].location == MemoryLocation::Gpu {
                    return "already_gpu".to_string();
                }
                let needed = self.pages[i].size_bytes;
                if self.total_gpu_bytes.saturating_add(needed) > self.gpu_budget {
                    let freed = self.evict_lru(needed);
                    if freed < needed {
                        return format!("need_gpu:{}_freed:{}", needed - freed, freed);
                    }
                }
                self.pages[i].location = MemoryLocation::Gpu;
                self.total_gpu_bytes += needed;
                "promoted".to_string()
            }
        }
    }

    pub fn demote_to_cpu(&mut self, id: u64) -> String {
        for page in &mut self.pages {
            if page.id == id {
                if page.location == MemoryLocation::Cpu {
                    return "already_cpu".to_string();
                }
                self.total_gpu_bytes -= page.size_bytes;
                page.location = MemoryLocation::Cpu;
                return "demoted".to_string();
            }
        }
        "absent".to_string()
    }

    pub fn get_best_candidates(&self, n: usize) -> Vec<u64> {
        let mut candidates: Vec<&MemoryPage> = self.pages.iter()
            .filter(|p| p.location == MemoryLocation::Gpu).collect();
        candidates.sort_by(|a, b| {
            let sa = score_for_eviction(a, self.clock);
            let sb = score_for_eviction(b, self.clock);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.iter().take(n).map(|p| p.id).collect()
    }

    pub fn stats(&self) -> String {
        let gpu_pages = self.pages.iter().filter(|p| p.location == MemoryLocation::Gpu).count();
        let cpu_pages = self.pages.iter().filter(|p| p.location == MemoryLocation::Cpu).count();
        format!(
            "pages={} gpu={} cpu={} gpu_bytes={} budget={} capacity={}",
            self.pages.len(), gpu_pages, cpu_pages,
            self.total_gpu_bytes, self.gpu_budget, self.capacity,
        )
    }

    pub fn update_priority(&mut self, id: u64, new_priority: f32) -> bool {
        for page in &mut self.pages {
            if page.id == id {
                page.priority = new_priority;
                return true;
            }
        }
        false
    }

    pub fn remove_page(&mut self, id: u64) -> bool {
        let pos = self.pages.iter().position(|p| p.id == id);
        if let Some(i) = pos {
            if self.pages[i].location == MemoryLocation::Gpu {
                self.total_gpu_bytes -= self.pages[i].size_bytes;
            }
            self.pages.swap_remove(i);
            true
        } else {
            false
        }
    }
}

impl MemoryPageTable {
    fn evict_lru(&mut self, needed_bytes: u64) -> u64 {
        let mut freed: u64 = 0;
        let mut to_evict: Vec<usize> = Vec::new();
        let mut indices: Vec<usize> = (0..self.pages.len()).collect();
        indices.sort_by(|&a, &b| {
            let sa = score_for_eviction(&self.pages[a], self.clock);
            let sb = score_for_eviction(&self.pages[b], self.clock);
            sb.partial_cmp(&sa).unwrap_or(std::cmp::Ordering::Equal)
        });
        for &idx in &indices {
            if freed >= needed_bytes { break; }
            let page = &self.pages[idx];
            if page.location != MemoryLocation::Gpu { continue; }
            freed += page.size_bytes;
            to_evict.push(idx);
        }
        for &idx in &to_evict {
            self.pages[idx].location = MemoryLocation::Cpu;
            self.total_gpu_bytes -= self.pages[idx].size_bytes;
        }
        freed
    }
}

fn score_for_eviction(page: &MemoryPage, clock: u64) -> f32 {
    let clock_f = clock.max(1) as f32;
    let recency = page.last_access as f32 / clock_f;
    let freq = page.access_count as f32 / clock_f;
    let priority_norm = page.priority.max(0.0).min(1.0);
    0.5 * recency + 0.3 * freq + 0.2 * (1.0 - priority_norm)
}

#[derive(Clone, Debug)]
#[repr(C)]
struct CheckpointHeader {
    magic: [u8; 8],
    version: u32,
    num_tensors: u32,
    total_bytes: u64,
    step: u64,
    timestamp: u64,
}

const MAGIC: [u8; 8] = *b"AURLCKPT";
const HEADER_SIZE: u64 = std::mem::size_of::<CheckpointHeader>() as u64;

#[pyclass]
pub struct MmapCheckpointWriter {
    path: String,
    index: Vec<(String, u64, u64)>,  // (name, offset, size)
    file: Option<File>,
    step: u64,
}

#[pymethods]
impl MmapCheckpointWriter {
    #[new]
    pub fn new(path: String) -> Self {
        MmapCheckpointWriter { path, index: Vec::new(), file: None, step: 0 }
    }

    pub fn open(&mut self) -> PyResult<()> {
        let file = OpenOptions::new()
            .create(true).read(true).write(true).truncate(true)
            .open(&self.path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
        let header = CheckpointHeader {
            magic: MAGIC, version: 1, num_tensors: 0,
            total_bytes: HEADER_SIZE, step: 0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs(),
        };
        let hp: *const CheckpointHeader = &header;
        let hb: &[u8] = unsafe { std::slice::from_raw_parts(hp as *const u8, HEADER_SIZE as usize) };
        (&file).write_all(hb)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
        file.set_len(HEADER_SIZE)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
        self.file = Some(file);
        Ok(())
    }

    pub fn write_tensor(&mut self, name: &str, data: Vec<f32>) -> PyResult<()> {
        let f = self.file.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("File not opened")
        })?;
        let offset = f.metadata()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?.len();
        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };
        let mut f = self.file.as_ref().unwrap();
        f.seek(SeekFrom::End(0))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
        f.write_all(bytes)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
        self.index.push((name.to_string(), offset, (data.len() * 4) as u64));
        self.step += 1;
        Ok(())
    }

    pub fn finalize(&mut self) -> PyResult<()> {
        let f = self.file.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("File not opened")
        })?;
        let total_bytes = f.metadata()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?.len();
        let header = CheckpointHeader {
            magic: MAGIC, version: 1, num_tensors: self.index.len() as u32,
            total_bytes, step: self.step,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs(),
        };
        let hp: *const CheckpointHeader = &header;
        let hb: &[u8] = unsafe { std::slice::from_raw_parts(hp as *const u8, HEADER_SIZE as usize) };
        let mut f = self.file.as_ref().unwrap();
        f.seek(SeekFrom::Start(0))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
        f.write_all(hb)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
        let index_bytes = serialize_index(&self.index);
        f.seek(SeekFrom::End(0))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
        f.write_all(&(index_bytes.len() as u64).to_le_bytes())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
        f.write_all(&index_bytes)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
        f.flush().map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
        Ok(())
    }

    pub fn get_checkpoint_size(&self) -> u64 {
        self.index.iter().map(|(_, _, s)| s).sum::<u64>() + HEADER_SIZE
    }
}

fn serialize_index(index: &[(String, u64, u64)]) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(index.len() as u32).to_le_bytes());
    for (name, offset, size) in index {
        let nb = name.as_bytes();
        bytes.extend_from_slice(&(nb.len() as u32).to_le_bytes());
        bytes.extend_from_slice(nb);
        bytes.extend_from_slice(&offset.to_le_bytes());
        bytes.extend_from_slice(&size.to_le_bytes());
    }
    bytes
}

#[pyclass]
pub struct DifferentialCheckpointer {
    base_path: String,
    dirty: Vec<String>,
}

#[pymethods]
impl DifferentialCheckpointer {
    #[new]
    pub fn new(base_path: String) -> Self {
        DifferentialCheckpointer { base_path, dirty: Vec::new() }
    }

    pub fn mark_dirty(&mut self, name: &str) {
        if !self.dirty.contains(&name.to_string()) {
            self.dirty.push(name.to_string());
        }
    }

    pub fn save_differential(&self, data: Vec<(String, Vec<f32>)>) -> PyResult<String> {
        let diff_path = format!("{}.diff", self.base_path);
        let file = OpenOptions::new()
            .create(true).write(true).truncate(true)
            .open(&diff_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
        let mut written = 0u32;
        for (name, tensor) in &data {
            if !self.dirty.contains(name) { continue; }
            let nb = name.as_bytes();
            let mut f = &file;
            f.write_all(&(nb.len() as u32).to_le_bytes())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            f.write_all(nb)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            f.write_all(&((tensor.len() * 4) as u64).to_le_bytes())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(tensor.as_ptr() as *const u8, tensor.len() * 4)
            };
            f.write_all(bytes)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            written += 1;
        }
        Ok(format!("Diff ckpt: {} dirty tensors -> {}", written, diff_path))
    }
}

#[pyfunction]
pub fn estimate_layer_memory(
    d_model: usize, d_ff: usize, n_heads: usize,
    seq_len: usize, batch_size: usize, precision_bytes: usize,
) -> String {
    let attn_params = 4 * d_model * d_model;
    let ffn_params = 3 * d_model * d_ff;
    let norm_params = 2 * d_model;
    let total_params = attn_params + ffn_params + norm_params;
    let param_bytes = total_params * precision_bytes;
    let act_per_token = (d_model + d_ff + n_heads * seq_len) * precision_bytes;
    let total_act = act_per_token * seq_len * batch_size;
    let total_mb = (param_bytes + total_act) as f64 / (1048576.0);
    format!(
        "layer: {}M params, {:.1}MB weights, {:.1}MB activations, total {:.1}MB",
        total_params / 1000000,
        param_bytes as f64 / 1048576.0,
        total_act as f64 / 1048576.0,
        total_mb,
    )
}

#[pymodule]
fn aurelius_memory(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MemoryPageTable>()?;
    m.add_class::<MmapCheckpointWriter>()?;
    m.add_class::<DifferentialCheckpointer>()?;
    m.add_function(wrap_pyfunction!(estimate_layer_memory, m)?)?;
    Ok(())
}
