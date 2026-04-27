use napi_derive::napi;
use regex::Regex;
use unicode_segmentation::UnicodeSegmentation;

#[napi(object)]
pub struct ChunkResult {
    pub chunks: Vec<String>,
    pub total_chunks: u32,
    pub total_characters: u32,
    pub strategy: String,
}

#[napi(object)]
pub struct TextStats {
    pub characters: u32,
    pub words: u32,
    pub sentences: u32,
    pub lines: u32,
    pub paragraphs: u32,
    pub graphemes: u32,
    pub avg_word_length: f64,
    pub reading_time_seconds: f64,
}

#[napi(object)]
pub struct SplitOptions {
    pub max_chunk_size: u32,
    pub overlap: u32,
    pub respect_sentences: bool,
    pub respect_paragraphs: bool,
}

#[napi(object)]
pub struct SentenceInfo {
    pub index: u32,
    pub text: String,
    pub char_start: u32,
    pub char_end: u32,
    pub word_count: u32,
}

/// Chunk text by token count (approximate, using word count as proxy)
fn chunk_by_tokens(text: &str, max_tokens: u32, overlap: u32) -> Vec<String> {
    let words: Vec<&str> = text.split_whitespace().collect();
    let max = max_tokens as usize;
    let ov = overlap as usize;
    if words.len() <= max {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start = 0;
    while start < words.len() {
        let end = (start + max).min(words.len());
        let chunk = words[start..end].join(" ");
        chunks.push(chunk);
        if end >= words.len() {
            break;
        }
        start = end.saturating_sub(ov);
    }
    chunks
}

/// Chunk text by character count with optional sentence/paragraph boundary awareness
fn chunk_by_chars(text: &str, max_chars: u32, overlap: u32, respect_sentences: bool, respect_paragraphs: bool) -> Vec<String> {
    let max = max_chars as usize;
    let ov = overlap as usize;
    if text.len() <= max {
        return vec![text.to_string()];
    }

    let mut chunks = Vec::new();
    let mut start = 0;

    while start < text.len() {
        let mut end = (start + max).min(text.len());

        if respect_paragraphs && end < text.len() {
            if let Some(para_end) = text[start..end].rfind("\n\n") {
                end = start + para_end + 2;
            }
        }

        if respect_sentences && end < text.len() {
            if let Some(sent_end) = text[start..end].rfind(|c| c == '.' || c == '!' || c == '?') {
                end = start + sent_end + 1;
            }
        }

        let chunk = &text[start..end];
        if !chunk.trim().is_empty() {
            chunks.push(chunk.to_string());
        }

        if end >= text.len() {
            break;
        }

        start = end.saturating_sub(ov);
    }

    chunks
}

#[napi]
pub struct TextProcessor {
    sentence_re: Regex,
    paragraph_re: Regex,
}

#[napi]
impl TextProcessor {
    #[napi(constructor)]
    pub fn new() -> Self {
        TextProcessor {
            sentence_re: Regex::new(r"[.!?]+[\s]+").unwrap(),
            paragraph_re: Regex::new(r"\n\s*\n").unwrap(),
        }
    }

    #[napi]
    pub fn stats(&self, text: String) -> TextStats {
        let words: Vec<&str> = text.split_whitespace().collect();
        let sentences: Vec<&str> = self.sentence_re.split(&text).collect();
        let paragraphs: Vec<&str> = self.paragraph_re.split(&text).collect();

        let avg_word_len = if words.is_empty() {
            0.0
        } else {
            words.iter().map(|w| w.len() as f64).sum::<f64>() / words.len() as f64
        };

        // Average reading speed: 238 words per minute
        let reading_time = words.len() as f64 / 238.0 * 60.0;

        TextStats {
            characters: text.chars().count() as u32,
            words: words.len() as u32,
            sentences: sentences.len() as u32,
            lines: text.lines().count() as u32,
            paragraphs: paragraphs.len() as u32,
            graphemes: text.graphemes(true).count() as u32,
            avg_word_length: (avg_word_len * 100.0).round() / 100.0,
            reading_time_seconds: (reading_time * 100.0).round() / 100.0,
        }
    }

    #[napi]
    pub fn chunk(&self, text: String, max_size: u32, overlap: u32, strategy: String) -> ChunkResult {
        let chunks = match strategy.as_str() {
            "token" => chunk_by_tokens(&text, max_size, overlap),
            "sentence" => self.chunk_by_sentences(&text, max_size as usize),
            "paragraph" => self.chunk_by_paragraphs(&text, max_size as usize),
            _ => chunk_by_chars(&text, max_size, overlap, false, false),
        };

        let total_chars: usize = chunks.iter().map(|c| c.len()).sum();

        ChunkResult {
            total_chunks: chunks.len() as u32,
            total_characters: total_chars as u32,
            strategy,
            chunks,
        }
    }

    #[napi]
    pub fn chunk_advanced(&self, text: String, options: SplitOptions) -> ChunkResult {
        let chunks = chunk_by_chars(
            &text,
            options.max_chunk_size,
            options.overlap,
            options.respect_sentences,
            options.respect_paragraphs,
        );

        let total_chars: usize = chunks.iter().map(|c| c.len()).sum();

        ChunkResult {
            total_chunks: chunks.len() as u32,
            total_characters: total_chars as u32,
            strategy: "advanced".to_string(),
            chunks,
        }
    }

    fn chunk_by_sentences(&self, text: &str, max_sentences: usize) -> Vec<String> {
        let sentences: Vec<&str> = self.sentence_re.split(text).collect();
        if sentences.len() <= max_sentences {
            return vec![text.to_string()];
        }

        let mut chunks = Vec::new();
        for chunk in sentences.chunks(max_sentences) {
            chunks.push(chunk.join(". ") + ".");
        }
        chunks
    }

    fn chunk_by_paragraphs(&self, text: &str, max_paragraphs: usize) -> Vec<String> {
        let paragraphs: Vec<&str> = self.paragraph_re.split(text).collect();
        if paragraphs.len() <= max_paragraphs {
            return vec![text.to_string()];
        }

        let mut chunks = Vec::new();
        for chunk in paragraphs.chunks(max_paragraphs) {
            chunks.push(chunk.join("\n\n"));
        }
        chunks
    }

    #[napi]
    pub fn split_sentences(&self, text: String) -> Vec<SentenceInfo> {
        let mut sentences = Vec::new();
        let mut char_pos = 0u32;

        // Use a more robust sentence splitting approach
        let mut current = String::new();
        let mut current_start = 0u32;

        for grapheme in text.graphemes(true) {
            current.push_str(grapheme);

            if grapheme == "." || grapheme == "!" || grapheme == "?" {
                let sentence_text = current.trim().to_string();
                if !sentence_text.is_empty() {
                    let word_count = sentence_text.split_whitespace().count() as u32;
                    sentences.push(SentenceInfo {
                        index: sentences.len() as u32,
                        text: sentence_text,
                        char_start: current_start,
                        char_end: char_pos + 1,
                        word_count,
                    });
                }
                current.clear();
                current_start = char_pos + 1;
            }

            char_pos += 1;
        }

        // Don't forget the last sentence if no punctuation
        let remaining = current.trim();
        if !remaining.is_empty() {
            let word_count = remaining.split_whitespace().count() as u32;
            sentences.push(SentenceInfo {
                index: sentences.len() as u32,
                text: remaining.to_string(),
                char_start: current_start,
                char_end: char_pos,
                word_count,
            });
        }

        sentences
    }

    #[napi]
    pub fn extract_keywords(&self, text: String, max_keywords: Option<u32>) -> Vec<String> {
        let max = max_keywords.unwrap_or(10) as usize;
        let stop_words = [
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for", "of", "by",
            "with", "from", "is", "was", "are", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "could", "should", "may", "might",
            "shall", "can", "need", "dare", "ought", "used", "this", "that", "these", "those",
            "i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them",
            "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs",
        ];

        let mut word_counts: Vec<(String, u32)> = Vec::new();

        for word in text.unicode_words() {
            let lower = word.to_lowercase();
            if lower.len() < 2 || stop_words.contains(&lower.as_str()) {
                continue;
            }

            if let Some((_, count)) = word_counts.iter_mut().find(|(w, _)| w == &lower) {
                *count += 1;
            } else {
                word_counts.push((lower, 1));
            }
        }

        word_counts.sort_by(|a, b| b.1.cmp(&a.1));
        word_counts.truncate(max);

        word_counts.into_iter().map(|(w, _)| w).collect()
    }

    #[napi]
    pub fn summarize(&self, text: String, max_sentences: u32) -> String {
        let sentences = self.split_sentences(text);
        let max = max_sentences as usize;

        if sentences.len() <= max {
            return sentences.iter().map(|s| s.text.clone()).collect::<Vec<_>>().join(" ");
        }

        // Simple extractive summarization: pick first and last sentences
        let mut selected = Vec::new();

        // First sentence (usually the topic sentence)
        if let Some(first) = sentences.first() {
            selected.push(first.text.clone());
        }

        // Pick sentences from middle with highest keyword density
        let keywords = self.extract_keywords(
            sentences.iter().map(|s| s.text.clone()).collect::<Vec<_>>().join(" "),
            Some(5),
        );

        let mut scored_sentences: Vec<(usize, u32)> = sentences
            .iter()
            .enumerate()
            .skip(1)
            .map(|(i, s)| {
                let score = keywords
                    .iter()
                    .filter(|kw| s.text.to_lowercase().contains(&kw.to_lowercase()))
                    .count() as u32;
                (i, score)
            })
            .collect();

        scored_sentences.sort_by(|a, b| b.1.cmp(&a.1));

        for &(idx, _) in scored_sentences.iter().take(max.saturating_sub(2)) {
            if !selected.contains(&sentences[idx].text) {
                selected.push(sentences[idx].text.clone());
            }
        }

        // Last sentence (usually the conclusion)
        if let Some(last) = sentences.last() {
            if !selected.contains(&last.text) {
                selected.push(last.text.clone());
            }
        }

        selected.join(" ")
    }

    #[napi]
    pub fn sliding_window(&self, text: String, window_size: u32, stride: u32) -> Vec<String> {
        let words: Vec<&str> = text.split_whitespace().collect();
        let ws = window_size as usize;
        let st = stride as usize;
        if words.len() <= ws {
            return vec![text.to_string()];
        }

        let mut windows = Vec::new();
        let mut start = 0;
        while start + ws <= words.len() {
            let window = words[start..start + ws].join(" ");
            windows.push(window);
            start += st;
        }
        // Last window if we didn't land exactly
        if start < words.len() && start + ws > words.len() {
            let window = words[words.len().saturating_sub(ws)..].join(" ");
            if !windows.last().map_or(false, |w| *w == window) {
                windows.push(window);
            }
        }

        windows
    }

    #[napi]
    pub fn normalize_whitespace(&self, text: String) -> String {
        let re = Regex::new(r"\s+").unwrap();
        let trimmed = text.trim();
        re.replace_all(trimmed, " ").to_string()
    }

    #[napi]
    pub fn truncate(&self, text: String, max_chars: u32, ellipsis: Option<String>) -> String {
        let max = max_chars as usize;
        if text.len() <= max {
            return text;
        }

        let ell = ellipsis.unwrap_or_else(|| "...".to_string());
        let mut truncated = text.chars().take(max.saturating_sub(ell.len())).collect::<String>();
        truncated.push_str(&ell);
        truncated
    }
}
