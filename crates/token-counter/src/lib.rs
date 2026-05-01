use napi_derive::napi;
use unicode_segmentation::UnicodeSegmentation;

#[napi(object)]
pub struct TokenCountResult {
    pub characters: u32,
    pub words: u32,
    pub sentences: u32,
    pub estimated_tokens: u32,
    pub lines: u32,
    pub grapheme_clusters: u32,
}

#[napi(object)]
pub struct TokenCountOptions {
    pub approximate: Option<bool>,
    pub model_context: Option<u32>,
}

#[napi(object)]
pub struct TokenBudget {
    pub available: u32,
    pub used: u32,
    pub remaining: u32,
    pub fraction_used: f64,
}

const TOKENS_PER_CHAR: f64 = 0.25;
const TOKENS_PER_WORD: f64 = 1.3;

#[napi]
pub fn count_tokens(text: String, options: Option<TokenCountOptions>) -> TokenCountResult {
    let approximate = options.as_ref().and_then(|o| o.approximate).unwrap_or(true);

    let characters = text.chars().count() as u32;
    let words = text.split_whitespace().count() as u32;
    let sentences = text
        .split(|c: char| c == '.' || c == '!' || c == '?')
        .filter(|s| !s.trim().is_empty())
        .count() as u32;
    let lines = text.lines().count() as u32;
    let grapheme_clusters = text.graphemes(true).count() as u32;

    let estimated_tokens = if approximate {
        (words as f64 * TOKENS_PER_WORD) as u32
    } else {
        ((characters as f64 * TOKENS_PER_CHAR) + (words as f64 * TOKENS_PER_WORD * 0.5)) as u32
    };

    TokenCountResult {
        characters,
        words,
        sentences,
        estimated_tokens,
        lines,
        grapheme_clusters,
    }
}

#[napi]
pub fn compute_token_budget(
    text: String,
    max_tokens: u32,
    options: Option<TokenCountOptions>,
) -> TokenBudget {
    let result = count_tokens(text, options);
    let used = result.estimated_tokens;
    let available = max_tokens;
    let remaining = available.saturating_sub(used);
    let fraction_used = if available > 0 {
        used as f64 / available as f64
    } else {
        1.0
    };

    TokenBudget {
        available,
        used,
        remaining,
        fraction_used,
    }
}

#[napi]
pub fn truncate_to_budget(
    text: String,
    max_tokens: u32,
    options: Option<TokenCountOptions>,
) -> String {
    let budget = compute_token_budget(text.clone(), max_tokens, options);

    if budget.fraction_used <= 1.0 {
        return text;
    }

    let words: Vec<&str> = text.split_whitespace().collect();
    let target_words = (max_tokens as f64 / TOKENS_PER_WORD) as usize;
    let truncated: Vec<&str> = words.into_iter().take(target_words).collect();
    truncated.join(" ")
}

#[napi]
pub fn estimate_tokens_for_messages(messages: Vec<String>) -> u32 {
    let overhead = 4; // per-message overhead in tokens
    messages
        .iter()
        .map(|msg| {
            let words = msg.split_whitespace().count() as u32;
            (words as f64 * TOKENS_PER_WORD) as u32 + overhead
        })
        .sum()
}

#[napi]
pub fn format_token_count(count: u32) -> String {
    if count < 1000 {
        format!("{} tokens", count)
    } else if count < 1_000_000 {
        format!("{:.1}k tokens", count as f64 / 1000.0)
    } else {
        format!("{:.1}M tokens", count as f64 / 1_000_000.0)
    }
}
