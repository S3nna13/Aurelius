use std::collections::HashMap;
use std::sync::Arc;

use dashmap::DashMap;
use napi_derive::napi;
use radix_trie::Trie;
use rust_stemmers::{Algorithm, Stemmer};
use unicode_segmentation::UnicodeSegmentation;

#[napi(object)]
pub struct IndexedDocument {
    pub id: String,
    pub field: String,
    pub content: String,
}

#[napi(object)]
pub struct SearchResult {
    pub id: String,
    pub field: String,
    pub score: f64,
    pub snippet: String,
}

#[napi(object)]
pub struct IndexStats {
    pub documents: u32,
    pub terms: u32,
    pub fields: Vec<String>,
}

#[napi(object)]
pub struct Suggestion {
    pub text: String,
    pub score: u32,
}

fn tokenize(text: &str) -> Vec<String> {
    let stemmer = Stemmer::create(Algorithm::English);
    text.unicode_words()
        .map(|w| w.to_lowercase())
        .map(|w| stemmer.stem(&w).to_string())
        .filter(|w| w.len() > 1)
        .collect()
}

fn generate_snippet(text: &str, query: &str, window: usize) -> String {
    let text_lower: String = text.to_lowercase();
    let q_lower: String = query.to_lowercase();
    if let Some(byte_pos) = text_lower.find(&q_lower) {
        let char_pos = text_lower[..byte_pos].chars().count();
        let q_chars = q_lower.chars().count();
        let start = char_pos.saturating_sub(window);
        let end = (char_pos + q_chars + window).min(text.chars().count());
        let snippet: String = text.chars().skip(start).take(end - start).collect();
        let prefix = if start > 0 { "..." } else { "" };
        let suffix = if end < text.chars().count() { "..." } else { "" };
        format!("{}{}{}", prefix, snippet, suffix)
    } else {
        text.chars().take(window * 2).collect()
    }
}

#[napi]
pub struct SearchIndex {
    inverted_index: Arc<DashMap<String, HashMap<String, Vec<(String, f64)>>>>,
    doc_store: Arc<DashMap<String, HashMap<String, String>>>,
    doc_lengths: Arc<DashMap<String, HashMap<String, u32>>>,
    autocomplete_trie: Arc<DashMap<String, Trie<String, u32>>>,
    avg_doc_length: Arc<DashMap<String, f64>>,
    total_docs: Arc<DashMap<String, u32>>,
}

#[napi]
impl SearchIndex {
    #[napi(constructor)]
    pub fn new() -> Self {
        SearchIndex {
            inverted_index: Arc::new(DashMap::new()),
            doc_store: Arc::new(DashMap::new()),
            doc_lengths: Arc::new(DashMap::new()),
            autocomplete_trie: Arc::new(DashMap::new()),
            avg_doc_length: Arc::new(DashMap::new()),
            total_docs: Arc::new(DashMap::new()),
        }
    }

    #[napi]
    pub fn index_document(&self, id: String, field: String, content: String) {
        let tokens = tokenize(&content);
        let field_key = field.clone();

        if !self.doc_lengths.contains_key(&field_key) {
            self.doc_lengths.insert(field_key.clone(), HashMap::new());
            self.avg_doc_length.insert(field_key.clone(), 0.0);
            self.total_docs.insert(field_key.clone(), 0);
            self.autocomplete_trie.insert(field_key.clone(), Trie::new());
        }

        self.doc_lengths.get_mut(&field_key).unwrap().insert(id.clone(), tokens.len() as u32);

        let mut total: u32 = 0;
        let mut count: u32 = 0;
        if let Some(lengths) = self.doc_lengths.get(&field_key) {
            for (_, len) in lengths.iter() {
                total += len;
                count += 1;
            }
        }
        if count > 0 {
            self.avg_doc_length.insert(field_key.clone(), total as f64 / count as f64);
        }
        *self.total_docs.get_mut(&field_key).unwrap() = count;

        if !self.inverted_index.contains_key(&field_key) {
            self.inverted_index.insert(field_key.clone(), HashMap::new());
        }

        // Count term frequency in this doc
        let mut term_freq: HashMap<String, f64> = HashMap::new();
        for token in &tokens {
            *term_freq.entry(token.clone()).or_insert(0.0) += 1.0;
        }

        // Normalize by doc length for BM25
        let doc_len = tokens.len() as f64;
        for (term, freq) in &term_freq {
            let tf = freq / doc_len;
            let mut idx = self.inverted_index.get_mut(&field_key).unwrap();
            idx.entry(term.clone()).or_insert_with(Vec::new).push((id.clone(), tf));

            // Update autocomplete trie
            if let Some(mut trie) = self.autocomplete_trie.get_mut(&field_key) {
                for i in 1..=term.len().min(10) {
                    let prefix = term[..i].to_string();
                    let existing = *trie.get(&prefix).unwrap_or(&0);
                    trie.insert(prefix, existing + 1);
                }
            }
        }

        // Store raw content for snippet generation
        if !self.doc_store.contains_key(&field_key) {
            self.doc_store.insert(field_key.clone(), HashMap::new());
        }
        self.doc_store.get_mut(&field_key).unwrap().insert(id.clone(), content.clone());
    }

    #[napi]
    pub fn search(&self, query: String, field: String, limit: Option<u32>) -> Vec<SearchResult> {
        let limit = limit.unwrap_or(10) as usize;
        let tokens = tokenize(&query);
        if tokens.is_empty() { return vec![]; }

        let field_key = field;
        let index = match self.inverted_index.get(&field_key) {
            Some(idx) => idx,
            None => return vec![],
        };

        let avgdl = self.avg_doc_length.get(&field_key).map(|v| *v).unwrap_or(1.0);
        let total_docs = self.total_docs.get(&field_key).map(|v| *v).unwrap_or(1) as f64;
        let k1 = 1.2;
        let b = 0.75;

        // BM25 scoring
        let mut scores: HashMap<String, f64> = HashMap::new();

        let doc_lengths_map: HashMap<String, u32> = self.doc_lengths.get(&field_key)
            .map(|l| l.iter().map(|(k, &v)| (k.clone(), v)).collect())
            .unwrap_or_default();

        for token in &tokens {
            if let Some(postings) = index.get(token) {
                let n_q = postings.len() as f64;
                let idf = ((total_docs - n_q + 0.5) / (n_q + 0.5) + 1.0).ln();
                for (doc_id, tf) in postings {
                    let doc_len = doc_lengths_map.get(doc_id).map(|&l| l as f64).unwrap_or(1.0);
                    let score = idf * ((tf * (k1 + 1.0)) / (tf + k1 * (1.0 - b + b * (doc_len / avgdl))));
                    *scores.entry(doc_id.clone()).or_insert(0.0) += score;
                }
            }
        }

        // Sort by score and build results
        let mut results: Vec<(String, f64)> = scores.into_iter().collect();
        results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        results.truncate(limit);

        let doc_contents: HashMap<String, String> = self.doc_store.get(&field_key)
            .map(|docs| docs.iter().map(|(k, v)| (k.clone(), v.clone())).collect())
            .unwrap_or_default();

        results.into_iter().map(|(id, score)| {
            let content = doc_contents.get(&id).cloned().unwrap_or_default();
            SearchResult {
                id,
                field: field_key.clone(),
                score: (score * 1000.0).round() / 1000.0,
                snippet: generate_snippet(&content, &query, 60),
            }
        }).collect()
    }

    #[napi]
    pub fn suggest(&self, prefix: String, field: String, limit: Option<u32>) -> Vec<Suggestion> {
        let limit = limit.unwrap_or(5) as usize;
        let stemmed = tokenize(&prefix);
        if stemmed.is_empty() { return vec![]; }

        let field_key = field;
        let trie = match self.autocomplete_trie.get(&field_key) {
            Some(t) => t,
            None => return vec![],
        };

        let q = stemmed[0].clone();
        let mut suggestions: Vec<(String, u32)> = Vec::new();
        if let Some(&val) = trie.get(&q) {
            suggestions.push((q.clone(), val));
        }
        let prefix = q.clone();
        for _ in 0..limit {
            let extended = format!("{}{}", prefix, suggestions.len());
            if let Some(&val) = trie.get(&extended) {
                suggestions.push((extended, val));
            } else {
                break;
            }
        }

        suggestions.sort_by(|a, b| b.1.cmp(&a.1));
        suggestions.truncate(limit);

        suggestions.into_iter()
            .map(|(text, score)| Suggestion { text, score })
            .collect()
    }

    #[napi]
    pub fn delete_document(&self, id: String, field: String) -> bool {
        let field_key = field;
        let mut removed = false;

        if let Some(mut docs) = self.doc_store.get_mut(&field_key) {
            if docs.remove(&id).is_some() { removed = true; }
        }

        if let Some(mut lengths) = self.doc_lengths.get_mut(&field_key) {
            lengths.remove(&id);
        }

        if removed {
            // Recalculate avg doc length
            if let Some(lengths) = self.doc_lengths.get(&field_key) {
                let total: u32 = lengths.iter().map(|(_, l)| l).sum();
                let count = lengths.len() as f64;
                if count > 0.0 {
                    self.avg_doc_length.insert(field_key.clone(), total as f64 / count);
                }
            }
            if let Some(mut total) = self.total_docs.get_mut(&field_key) {
                *total = total.saturating_sub(1);
            }
        }

        removed
    }

    #[napi]
    pub fn clear(&self, field: Option<String>) {
        if let Some(f) = field {
            self.inverted_index.remove(&f);
            self.doc_store.remove(&f);
            self.doc_lengths.remove(&f);
            self.autocomplete_trie.remove(&f);
            self.avg_doc_length.remove(&f);
            self.total_docs.remove(&f);
        } else {
            self.inverted_index.clear();
            self.doc_store.clear();
            self.doc_lengths.clear();
            self.autocomplete_trie.clear();
            self.avg_doc_length.clear();
            self.total_docs.clear();
        }
    }

    #[napi]
    pub fn get_stats(&self) -> Vec<IndexStats> {
        let mut stats = vec![];
        let fields: Vec<String> = self.doc_store.iter().map(|e| e.key().clone()).collect();

        for field in fields {
            let docs = self.doc_store.get(&field).map(|d| d.len() as u32).unwrap_or(0);
            let terms = self.inverted_index.get(&field).map(|i| i.len() as u32).unwrap_or(0);
            let fields_list = vec![field.clone()];
            stats.push(IndexStats { documents: docs, terms, fields: fields_list });
        }

        stats
    }
}
