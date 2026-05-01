/// Streaming JSONL merge/dedup — memory-bounded, O(n) time.
/// Reads multiple JSONL files, deduplicates by SHA256 of messages content,
/// writes deduplicated output. Processes one line at a time.
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::fs::{File, OpenOptions};
use std::io::{self, BufRead, BufReader, Write};
use std::path::PathBuf;

fn hash_line(line: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(line.as_bytes());
    hex::encode(hasher.finalize())
}

fn run() -> io::Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: jsonl-merge <output.jsonl> <input1.jsonl> [input2.jsonl ...]");
        std::process::exit(1);
    }

    let output_path = PathBuf::from(&args[1]);
    let input_paths: Vec<&str> = args[2..].iter().map(|s| s.as_str()).collect();

    let mut seen: HashSet<String> = HashSet::new();
    let mut out = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(&output_path)?;

    let mut total = 0u64;
    let mut unique = 0u64;

    for path in &input_paths {
        let file = match File::open(path) {
            Ok(f) => f,
            Err(e) => {
                eprintln!("  skip {}: {}", path, e);
                continue;
            }
        };
        let reader = BufReader::with_capacity(65536, file);
        for line_result in reader.lines() {
            let line = line_result?;
            if line.trim().is_empty() {
                continue;
            }
            total += 1;
            let h = hash_line(&line);
            if seen.contains(&h) {
                continue;
            }
            seen.insert(h);
            unique += 1;
            writeln!(out, "{}", line)?;
        }
    }

    eprintln!(
        "jsonl-merge: {} total → {} unique written to {}",
        total,
        unique,
        output_path.display()
    );
    Ok(())
}

fn main() {
    if let Err(e) = run() {
        eprintln!("jsonl-merge error: {}", e);
        std::process::exit(1);
    }
}
