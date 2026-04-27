import { useState, useRef, useEffect, useCallback } from 'react'
import { Search, X, Loader2, Command, ArrowUp, ArrowDown } from 'lucide-react'
import { useDebounce } from '../hooks/useDebounce'
import { useKeyboardShortcut } from '../hooks/useKeyboard'
import { useClickOutside } from '../hooks/useClickOutside'

interface SearchResult {
  type: string
  label: string
  description?: string
  id?: string
  onClick?: () => void
}

interface SearchBarProps {
  placeholder?: string
  onSearch?: (query: string) => void
  results?: SearchResult[]
  loading?: boolean
  autoFocus?: boolean
}

export default function SearchBar({ placeholder = 'Search...', onSearch, results = [], loading }: SearchBarProps) {
  const [query, setQuery] = useState('')
  const [open, setOpen] = useState(false)
  const [selectedIdx, setSelectedIdx] = useState(-1)
  const inputRef = useRef<HTMLInputElement>(null)
  const debouncedQuery = useDebounce(query, 300)
  const panelRef = useClickOutside<HTMLDivElement>(() => setOpen(false))

  useEffect(() => {
    if (debouncedQuery && debouncedQuery.length >= 2) {
      onSearch?.(debouncedQuery)
      setOpen(true)
    } else {
      setOpen(false)
    }
  }, [debouncedQuery, onSearch])

  useKeyboardShortcut('k', () => {
    inputRef.current?.focus()
  }, [{ meta: true }])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (!open || results.length === 0) return
    if (e.key === 'ArrowDown') { e.preventDefault(); setSelectedIdx((i) => Math.min(i + 1, results.length - 1)) }
    else if (e.key === 'ArrowUp') { e.preventDefault(); setSelectedIdx((i) => Math.max(i - 1, 0)) }
    else if (e.key === 'Enter' && selectedIdx >= 0) { results[selectedIdx]?.onClick?.(); setOpen(false) }
    else if (e.key === 'Escape') setOpen(false)
  }, [open, results, selectedIdx])

  const typeColors: Record<string, string> = {
    agent: 'bg-[#4fc3f7]/10 text-[#4fc3f7] border-[#4fc3f7]/20',
    activity: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
    log: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
    memory: 'bg-purple-500/10 text-purple-400 border-purple-500/20',
    notification: 'bg-rose-500/10 text-rose-400 border-rose-500/20',
    config: 'bg-[#2d2d44]/30 text-[#9e9eb0] border-[#2d2d44]/40',
  }

  return (
    <div ref={panelRef} className="relative w-full">
      <div className="relative">
        <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#9e9eb0]" />
        <input
          ref={inputRef}
          type="text"
          value={query}
          onChange={(e) => { setQuery(e.target.value); setSelectedIdx(-1) }}
          onFocus={() => { if (results.length > 0) setOpen(true) }}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          className="w-full bg-[#0f0f1a] border border-[#2d2d44] rounded-lg pl-9 pr-8 py-2 text-sm text-[#e0e0e0] placeholder:text-[#9e9eb0] focus:outline-none focus:border-[#4fc3f7] transition-colors"
        />
        <div className="absolute right-3 top-1/2 -translate-y-1/2 flex items-center gap-1">
          {loading && <Loader2 size={12} className="animate-spin text-[#9e9eb0]" />}
          {query && <button onClick={() => { setQuery(''); setOpen(false); inputRef.current?.focus() }} className="text-[#9e9eb0] hover:text-[#e0e0e0]"><X size={14} /></button>}
          {!query && <kbd className="hidden sm:inline-flex text-[10px] text-[#9e9eb0]/40 bg-[#2d2d44]/30 px-1.5 py-0.5 rounded border border-[#2d2d44]/40"><Command size={10} />K</kbd>}
        </div>
      </div>

      {open && (
        <div className="absolute z-50 mt-1 w-full bg-[#1a1a2e] border border-[#2d2d44] rounded-lg shadow-2xl overflow-hidden">
          {results.length === 0 && debouncedQuery.length >= 2 && !loading ? (
            <div className="px-4 py-6 text-center text-sm text-[#9e9eb0]">No results for "{debouncedQuery}"</div>
          ) : (
            <div className="max-h-80 overflow-y-auto py-1">
              {results.map((r, i) => (
                <button
                  key={`${r.type}-${r.id || i}`}
                  onClick={() => { r.onClick?.(); setOpen(false) }}
                  onMouseEnter={() => setSelectedIdx(i)}
                  className={`w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors ${i === selectedIdx ? 'bg-[#4fc3f7]/10' : 'hover:bg-[#2d2d44]/30'}`}
                >
                  <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded border uppercase ${typeColors[r.type] || 'bg-[#2d2d44]/30 text-[#9e9eb0]'}`}>{r.type}</span>
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-[#e0e0e0] truncate">{r.label}</p>
                    {r.description && <p className="text-xs text-[#9e9eb0] truncate">{r.description}</p>}
                  </div>
                </button>
              ))}
            </div>
          )}
          {results.length > 0 && (
            <div className="px-4 py-2 border-t border-[#2d2d44]/50 flex items-center gap-3 text-[10px] text-[#9e9eb0]/40">
              <span><ArrowUp size={10} className="inline" /><ArrowDown size={10} className="inline" /> navigate</span>
              <span>↵ select</span>
              <span>esc close</span>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
