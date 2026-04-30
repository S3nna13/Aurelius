use napi_derive::napi;

#[napi(object)]
pub struct SimilarityResult {
    pub index: u32,
    pub score: f64,
}

#[napi(object)]
pub struct TopKResult {
    pub results: Vec<SimilarityResult>,
    pub method: String,
}

#[napi(object)]
pub struct VectorStats {
    pub dimensions: u32,
    pub magnitude: f64,
    pub min: f64,
    pub max: f64,
    pub mean: f64,
    pub variance: f64,
}

fn validate_vectors(a: &[f64], b: &[f64]) -> Result<(), String> {
    if a.is_empty() || b.is_empty() {
        return Err("Vectors must not be empty".to_string());
    }
    if a.len() != b.len() {
        return Err(format!("Dimension mismatch: {} vs {}", a.len(), b.len()));
    }
    Ok(())
}

#[napi]
pub fn cosine_similarity(a: Vec<f64>, b: Vec<f64>) -> f64 {
    if validate_vectors(&a, &b).is_err() {
        return 0.0;
    }

    let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
    let mag_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    (dot / (mag_a * mag_b)).clamp(-1.0, 1.0)
}

#[napi]
pub fn dot_product(a: Vec<f64>, b: Vec<f64>) -> f64 {
    if validate_vectors(&a, &b).is_err() {
        return 0.0;
    }

    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

#[napi]
pub fn euclidean_distance(a: Vec<f64>, b: Vec<f64>) -> f64 {
    if validate_vectors(&a, &b).is_err() {
        return 0.0;
    }

    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y) * (x - y))
        .sum::<f64>()
        .sqrt()
}

#[napi]
pub fn manhattan_distance(a: Vec<f64>, b: Vec<f64>) -> f64 {
    if validate_vectors(&a, &b).is_err() {
        return 0.0;
    }

    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).sum()
}

#[napi]
pub fn jaccard_similarity(set_a: Vec<f64>, set_b: Vec<f64>) -> f64 {
    if set_a.is_empty() || set_b.is_empty() {
        return 0.0;
    }

    const EPS: f64 = 1e-10;
    let intersection = set_a
        .iter()
        .filter(|x| set_b.iter().any(|y| (*x - *y).abs() < EPS))
        .count();
    let union_size = set_a.len() + set_b.len() - intersection;

    if union_size == 0 {
        return 0.0;
    }

    intersection as f64 / union_size as f64
}

#[napi]
pub fn top_k_cosine(query: Vec<f64>, candidates: Vec<Vec<f64>>, k: u32) -> TopKResult {
    let k = (k as usize).min(candidates.len());

    let mut scores: Vec<(usize, f64)> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| (i, cosine_similarity(query.clone(), c.clone())))
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(k);

    TopKResult {
        results: scores
            .into_iter()
            .map(|(i, s)| SimilarityResult {
                index: i as u32,
                score: (s * 10000.0).round() / 10000.0,
            })
            .collect(),
        method: "cosine".to_string(),
    }
}

#[napi]
pub fn top_k_dot_product(query: Vec<f64>, candidates: Vec<Vec<f64>>, k: u32) -> TopKResult {
    let k = (k as usize).min(candidates.len());

    let mut scores: Vec<(usize, f64)> = candidates
        .iter()
        .enumerate()
        .map(|(i, c)| (i, dot_product(query.clone(), c.clone())))
        .collect();

    scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scores.truncate(k);

    TopKResult {
        results: scores
            .into_iter()
            .map(|(i, s)| SimilarityResult {
                index: i as u32,
                score: (s * 10000.0).round() / 10000.0,
            })
            .collect(),
        method: "dot_product".to_string(),
    }
}

#[napi]
pub fn normalize_vector(vec: Vec<f64>) -> Vec<f64> {
    let mag: f64 = vec.iter().map(|x| x * x).sum::<f64>().sqrt();
    if mag == 0.0 {
        return vec;
    }
    vec.iter().map(|x| x / mag).collect()
}

#[napi]
pub fn vector_stats(vec: Vec<f64>) -> VectorStats {
    let n = vec.len() as f64;
    if n == 0.0 {
        return VectorStats {
            dimensions: 0,
            magnitude: 0.0,
            min: 0.0,
            max: 0.0,
            mean: 0.0,
            variance: 0.0,
        };
    }

    let min = vec.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = vec.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let sum: f64 = vec.iter().sum();
    let mean = sum / n;
    let variance = vec.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
    let magnitude = vec.iter().map(|x| x * x).sum::<f64>().sqrt();

    VectorStats {
        dimensions: vec.len() as u32,
        magnitude: (magnitude * 1000.0).round() / 1000.0,
        min,
        max,
        mean: (mean * 1000.0).round() / 1000.0,
        variance: (variance * 1000.0).round() / 1000.0,
    }
}

#[napi]
pub fn pairwise_similarity(vectors: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
    let n = vectors.len();
    let mut matrix = vec![vec![0.0; n]; n];

    for i in 0..n {
        for j in i..n {
            let sim = cosine_similarity(vectors[i].clone(), vectors[j].clone());
            matrix[i][j] = (sim * 10000.0).round() / 10000.0;
            matrix[j][i] = matrix[i][j];
        }
    }

    matrix
}

#[napi]
pub fn batch_cosine_similarity(query: Vec<f64>, candidates: Vec<Vec<f64>>) -> Vec<f64> {
    candidates
        .iter()
        .map(|c| {
            let sim = cosine_similarity(query.clone(), c.clone());
            (sim * 10000.0).round() / 10000.0
        })
        .collect()
}
