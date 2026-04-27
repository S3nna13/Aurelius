export interface ChunkResult { chunks: string[]; totalChunks: number; totalCharacters: number; strategy: string }
export interface TextStats { characters: number; words: number; sentences: number; lines: number; paragraphs: number; graphemes: number; avgWordLength: number; readingTimeSeconds: number }
export interface SplitOptions { maxChunkSize: number; overlap: number; respectSentences: boolean; respectParagraphs: boolean }
export interface SentenceInfo { index: number; text: string; charStart: number; charEnd: number; wordCount: number }

export declare class TextProcessor {
  constructor()
  stats(text: string): TextStats
  chunk(text: string, maxSize: number, overlap: number, strategy: string): ChunkResult
  chunkAdvanced(text: string, options: SplitOptions): ChunkResult
  splitSentences(text: string): SentenceInfo[]
  extractKeywords(text: string, maxKeywords?: number): string[]
  summarize(text: string, maxSentences: number): string
  slidingWindow(text: string, windowSize: number, stride: number): string[]
  normalizeWhitespace(text: string): string
  truncate(text: string, maxChars: number, ellipsis?: string): string
}
