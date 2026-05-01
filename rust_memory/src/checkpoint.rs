use pyo3::prelude::*;
use std::collections::HashMap;
use std::fs::{File, OpenOptions};
use std::io::{Read, Write, Seek, SeekFrom};
use std::mem::size_of;
use std::path::Path;
use std::sync::Mutex;

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
const HEADER_SIZE: u64 = size_of::<CheckpointHeader>() as u64;

#[pyclass]
pub struct MmapCheckpointWriter {
    path: String,
    index: HashMap<String, (u64, u64)>,  // name -> (offset, size)
    file: Option<File>,
    step: u64,
}

#[pymethods]
impl MmapCheckpointWriter {
    #[new]
    pub fn new(path: String) -> Self {
        MmapCheckpointWriter {
            path,
            index: HashMap::new(),
            file: None,
            step: 0,
        }
    }

    pub fn open(&mut self) -> PyResult<()> {
        let file = OpenOptions::new()
            .create(true)
            .read(true)
            .write(true)
            .truncate(true)
            .open(&self.path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;

        let header = CheckpointHeader {
            magic: MAGIC,
            version: 1,
            num_tensors: 0,
            total_bytes: HEADER_SIZE,
            step: 0,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        let header_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(&header as *const _ as *const u8, size_of::<CheckpointHeader>()) };

        let mut f = &file;
        f.write_all(header_bytes)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;

        file.set_len(HEADER_SIZE)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;

        self.file = Some(file);
        Ok(())
    }

    pub fn write_tensor(&mut self, name: &str, data: Vec<f32>) -> PyResult<()> {
        let file = self.file.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("File not opened")
        })?;

        let offset = file.metadata()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?
            .len();

        let bytes: &[u8] = unsafe {
            std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4)
        };

        let mut file_mut = self.file.as_ref().unwrap();
        file_mut.seek(SeekFrom::End(0))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;

        file_mut.write_all(bytes)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;

        let size = (data.len() * 4) as u64;
        self.index.insert(name.to_string(), (offset, size));
        self.step += 1;
        Ok(())
    }

    pub fn finalize(&mut self) -> PyResult<()> {
        let file = self.file.as_ref().ok_or_else(|| {
            PyErr::new::<pyo3::exceptions::PyRuntimeError, _>("File not opened")
        })?;

        let total_bytes = file.metadata()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?
            .len();

        let header = CheckpointHeader {
            magic: MAGIC,
            version: 1,
            num_tensors: self.index.len() as u32,
            total_bytes,
            step: self.step,
            timestamp: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs(),
        };

        let header_bytes: &[u8] =
            unsafe { std::slice::from_raw_parts(&header as *const _ as *const u8, size_of::<CheckpointHeader>()) };

        let mut file_mut = self.file.as_ref().unwrap();
        file_mut.seek(SeekFrom::Start(0))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
        file_mut.write_all(header_bytes)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;

        let offset = total_bytes;
        let index_bytes = bincode_index(&self.index);
        file_mut.seek(SeekFrom::End(0))
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
        file_mut.write_all(&(index_bytes.len() as u64).to_le_bytes())
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
        file_mut.write_all(&index_bytes)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;

        file_mut.flush()
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;

        println!("Checkpoint written: {} tensors, {} bytes",
                 self.index.len(), total_bytes);
        Ok(())
    }

    pub fn get_checkpoint_size(&self) -> u64 {
        self.index.values().map(|(_, size)| size).sum::<u64>() + HEADER_SIZE
    }
}

fn bincode_index(index: &HashMap<String, (u64, u64)>) -> Vec<u8> {
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(index.len() as u32).to_le_bytes());
    for (name, (offset, size)) in index {
        let name_bytes = name.as_bytes();
        bytes.extend_from_slice(&(name_bytes.len() as u32).to_le_bytes());
        bytes.extend_from_slice(name_bytes);
        bytes.extend_from_slice(&offset.to_le_bytes());
        bytes.extend_from_slice(&size.to_le_bytes());
    }
    bytes
}

#[pyclass]
pub struct DifferentialCheckpointer {
    base_path: String,
    base_checkpoint: Option<HashMap<String, Vec<f32>>>,
    dirty_pages: HashMap<String, bool>,
}

#[pymethods]
impl DifferentialCheckpointer {
    #[new]
    pub fn new(base_path: String) -> Self {
        DifferentialCheckpointer {
            base_path,
            base_checkpoint: None,
            dirty_pages: HashMap::new(),
        }
    }

    pub fn mark_dirty(&mut self, name: &str) {
        self.dirty_pages.insert(name.to_string(), true);
    }

    pub fn save_differential(&self, data: Vec<(String, Vec<f32>)>) -> PyResult<String> {
        let diff_path = format!("{}.diff", self.base_path);
        let file = OpenOptions::new()
            .create(true)
            .write(true)
            .truncate(true)
            .open(&diff_path)
            .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;

        let mut written: u32 = 0;
        for (name, tensor) in &data {
            if !self.dirty_pages.get(name).unwrap_or(&false) {
                continue;
            }
            let name_bytes = name.as_bytes();
            let header_bytes = (name_bytes.len() as u32).to_le_bytes();
            let tensor_size = (tensor.len() * 4) as u64;

            let mut f = &file;
        f.write_all(&header_bytes)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            let mut f = &file;
        f.write_all(name_bytes)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            let mut f = &file;
        f.write_all(&tensor_size.to_le_bytes())
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;

            let bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(tensor.as_ptr() as *const u8, tensor.len() * 4)
            };
            let mut f = &file;
        f.write_all(bytes)
                .map_err(|e| PyErr::new::<pyo3::exceptions::PyIOError, _>(format!("{}", e)))?;
            written += 1;
        }

        Ok(format!("Differential checkpoint: {} dirty tensors saved to {}", written, diff_path))
    }
}

#[pymodule]
fn aurelius_checkpoint(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_class::<MmapCheckpointWriter>()?;
    m.add_class::<DifferentialCheckpointer>()?;
    Ok(())
}
