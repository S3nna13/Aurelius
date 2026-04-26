export interface TokenCountResult {
  characters: number
  words: number
  sentences: number
  estimatedTokens: number
  lines: number
  graphemeClusters: number
}

export interface TokenCountOptions {
  approximate?: boolean
  modelContext?: number
}

export interface TokenBudget {
  available: number
  used: number
  remaining: number
  fractionUsed: number
}

export function countTokens(text: string, options?: TokenCountOptions): TokenCountResult
export function computeTokenBudget(text: string, maxTokens: number, options?: TokenCountOptions): TokenBudget
export function truncateToBudget(text: string, maxTokens: number, options?: TokenCountOptions): string
export function estimateTokensForMessages(messages: string[]): number
export function formatTokenCount(count: number): string
