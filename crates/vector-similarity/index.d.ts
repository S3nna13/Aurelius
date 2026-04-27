export interface SimilarityResult { index: number; score: number }
export interface TopKResult { results: SimilarityResult[]; method: string }
export interface VectorStats { dimensions: number; magnitude: number; min: number; max: number; mean: number; variance: number }

export function cosineSimilarity(a: number[], b: number[]): number
export function dotProduct(a: number[], b: number[]): number
export function euclideanDistance(a: number[], b: number[]): number
export function manhattanDistance(a: number[], b: number[]): number
export function jaccardSimilarity(setA: number[], setB: number[]): number
export function topKCosine(query: number[], candidates: number[][], k: number): TopKResult
export function topKDotProduct(query: number[], candidates: number[][], k: number): TopKResult
export function normalizeVector(vec: number[]): number[]
export function vectorStats(vec: number[]): VectorStats
export function pairwiseSimilarity(vectors: number[][]): number[][]
export function batchCosineSimilarity(query: number[], candidates: number[][]): number[]
