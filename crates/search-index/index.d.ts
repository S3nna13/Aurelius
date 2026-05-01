export interface IndexedDocument { id: string; field: string; content: string }
export interface SearchResult { id: string; field: string; score: number; snippet: string }
export interface IndexStats { documents: number; terms: number; fields: string[] }
export interface Suggestion { text: string; score: number }

export declare class SearchIndex {
  constructor()
  indexDocument(id: string, field: string, content: string): void
  search(query: string, field: string, limit?: number): SearchResult[]
  suggest(prefix: string, field: string, limit?: number): Suggestion[]
  deleteDocument(id: string, field: string): boolean
  clear(field?: string): void
  getStats(): IndexStats[]
}
