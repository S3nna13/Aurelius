import { useState, useEffect, useCallback } from 'react'
import { Brain, Layers, Plus, Loader2, RefreshCw, Search, Clock, TrendingUp } from 'lucide-react'
import { useToast } from '../components/ToastProvider'

interface MemoryLayer { name: string; entries: number }
interface MemoryEntry { id: string; content: string; layer: string; timestamp: string; accessCount: number; importanceScore: number }

export default function Memory() {
  const { toast } = useToast()
  const [layers, setLayers] = useState<MemoryLayer[]>([])
  const [entries, setEntries] = useState<MemoryEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [selectedLayer, setSelectedLayer] = useState('')
  const [searchQuery, setSearchQuery] = useState('')
  const [newEntryLayer, setNewEntryLayer] = useState('')
  const [newEntryContent, setNewEntryContent] = useState('')
  const [adding, setAdding] = useState(false)

  const fetchData = useCallback(async () => {
    setLoading(true)
    try {
      const params = new URLSearchParams()
      if (selectedLayer) params.set('layer', selectedLayer)
      if (searchQuery) params.set('query', searchQuery)
      const [layersRes, entriesRes] = await Promise.all([
        fetch('/api/memory/layers'),
        fetch(`/api/memory/entries?${params}`),
      ])
      if (layersRes.ok) setLayers((await layersRes.json()).layers || [])
      if (entriesRes.ok) setEntries((await entriesRes.json()).entries || [])
    } catch { /* ignore */ }
    setLoading(false)
  }, [selectedLayer, searchQuery])

  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => { fetchData() }, [fetchData])

  const addEntry = async () => {
    if (!newEntryLayer || !newEntryContent.trim()) return
    setAdding(true)
    try {
      const res = await fetch('/api/memory/entries', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ layer: newEntryLayer, content: newEntryContent.trim() }),
      })
      const data = await res.json()
      if (data.success) {
        toast('Memory entry added', 'success')
        setNewEntryContent('')
        fetchData()
      }
    } catch { toast('Failed to add entry', 'error') }
    setAdding(false)
  }

  return (
    <div className="space-y-6 max-w-5xl">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2"><Brain size={20} className="text-[#4fc3f7]" /> Memory</h2>
        <button onClick={fetchData} disabled={loading} className="aurelius-btn-outline flex items-center gap-1.5 text-sm disabled:opacity-50"><RefreshCw size={14} /> Refresh</button>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-1 space-y-4">
          <div className="aurelius-card space-y-3">
            <h3 className="text-sm font-semibold text-[#e0e0e0] uppercase tracking-wider flex items-center gap-2"><Layers size={16} className="text-[#4fc3f7]" /> Layers</h3>
            <div className="space-y-1">
              <button onClick={() => setSelectedLayer('')} className={`w-full text-left px-3 py-2 rounded-lg text-sm transition-colors ${!selectedLayer ? 'bg-[#4fc3f7]/10 text-[#4fc3f7]' : 'text-[#9e9eb0] hover:text-[#e0e0e0] hover:bg-[#0f0f1a]/50'}`}>All Layers</button>
              {layers.map((layer) => (
                <button key={layer.name} onClick={() => setSelectedLayer(layer.name)}
                  className={`w-full flex items-center justify-between px-3 py-2 rounded-lg text-sm transition-colors ${selectedLayer === layer.name ? 'bg-[#4fc3f7]/10 text-[#4fc3f7]' : 'text-[#9e9eb0] hover:text-[#e0e0e0] hover:bg-[#0f0f1a]/50'}`}>
                  <span>{layer.name}</span>
                  <span className="text-[10px] opacity-60">{layer.entries}</span>
                </button>
              ))}
            </div>
          </div>

          <div className="aurelius-card space-y-3">
            <h3 className="text-sm font-semibold text-[#e0e0e0] uppercase tracking-wider flex items-center gap-2"><Plus size={16} className="text-[#4fc3f7]" /> Add Entry</h3>
            <select value={newEntryLayer} onChange={(e) => setNewEntryLayer(e.target.value)} className="w-full bg-[#0f0f1a] border border-[#2d2d44] rounded-lg px-3 py-2 text-sm text-[#e0e0e0] focus:outline-none focus:border-[#4fc3f7]">
              <option value="">Select layer...</option>
              {layers.map((l) => <option key={l.name} value={l.name}>{l.name}</option>)}
            </select>
            <textarea value={newEntryContent} onChange={(e) => setNewEntryContent(e.target.value)} placeholder="Memory content..." rows={3}
              className="w-full bg-[#0f0f1a] border border-[#2d2d44] rounded-lg px-3 py-2 text-sm text-[#e0e0e0] placeholder:text-[#9e9eb0] focus:outline-none focus:border-[#4fc3f7] resize-none" />
            <button onClick={addEntry} disabled={adding || !newEntryLayer || !newEntryContent.trim()} className="w-full aurelius-btn text-sm disabled:opacity-50">{adding ? <Loader2 size={14} className="animate-spin" /> : 'Add Entry'}</button>
          </div>
        </div>

        <div className="lg:col-span-3 space-y-4">
          <div className="relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#9e9eb0]" />
            <input type="text" placeholder="Search memory entries..." value={searchQuery} onChange={(e) => setSearchQuery(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') fetchData() }}
              className="w-full bg-[#0f0f1a] border border-[#2d2d44] rounded-lg pl-9 pr-3 py-2 text-sm text-[#e0e0e0] placeholder:text-[#9e9eb0] focus:outline-none focus:border-[#4fc3f7]" />
          </div>

          {loading ? (
            <div className="aurelius-card text-center py-12 text-[#9e9eb0]"><Loader2 size={24} className="mx-auto mb-2 animate-spin opacity-60" /><p>Loading memory...</p></div>
          ) : entries.length === 0 ? (
            <div className="aurelius-card text-center py-12 text-[#9e9eb0]"><Brain size={32} className="mx-auto mb-2 opacity-40" /><p>No memory entries found.</p></div>
          ) : (
            <div className="space-y-2">
              {entries.map((entry) => (
                <div key={entry.id} className="aurelius-card">
                  <div className="flex items-start justify-between gap-4">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm text-[#e0e0e0]">{entry.content}</p>
                      <div className="flex items-center gap-3 mt-2 text-[10px] text-[#9e9eb0]/60">
                        <span className="flex items-center gap-1"><Layers size={10} />{entry.layer}</span>
                        <span className="flex items-center gap-1"><Clock size={10} />{entry.timestamp}</span>
                        <span className="flex items-center gap-1"><TrendingUp size={10} />{entry.importanceScore.toFixed(2)}</span>
                        <span>Access: {entry.accessCount}</span>
                      </div>
                    </div>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
