import { useState, useEffect, useCallback } from 'react'
import { Cpu, Play, Square, RefreshCw, Loader2, AlertTriangle, Info, Zap } from 'lucide-react'
import { api } from '../api/AureliusClient'
import type { ModelInfo } from '../api/types'

const stateColors: Record<string, string> = {
  loaded: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
  loading: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
  unloaded: 'text-[#9e9eb0] bg-[#2d2d44]/20 border-[#2d2d44]/40',
  error: 'text-rose-400 bg-rose-500/10 border-rose-500/20',
}

function fmtParams(n: number): string {
  if (n >= 1_000_000_000) return `${(n / 1_000_000_000).toFixed(1)}B`
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`
  return `${n}`
}

export default function Models() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchModels = useCallback(async () => {
    try {
      const res: any = await api.get('/models')
      setModels(res.models || [])
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load models')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetchModels() }, [fetchModels])

  const toggleModel = async (id: string, currentState: string) => {
    const newState = currentState === 'loaded' ? 'unloaded' : 'loading'
    try {
      await api.post(`/models/${id}/state`, { state: newState })
      if (newState === 'loading') {
        setTimeout(async () => {
          await api.post(`/models/${id}/state`, { state: 'loaded' })
          fetchModels()
        }, 1500)
      }
      fetchModels()
    } catch { /* ignore */ }
  }

  const loadedCount = models.filter((m: any) => m.state === 'loaded').length

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
          <Cpu size={20} className="text-[#4fc3f7]" /> Model Manager
        </h2>
        <div className="flex items-center gap-3">
          <span className="text-xs text-[#9e9eb0]">{loadedCount}/{models.length} loaded</span>
          <button onClick={fetchModels} disabled={loading} className="aurelius-btn-outline flex items-center gap-1.5 text-sm disabled:opacity-50">
            {loading ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />} Refresh
          </button>
        </div>
      </div>

      {error && <div className="aurelius-card border-rose-500/30 bg-rose-500/5 text-rose-300 text-sm flex items-center gap-2"><AlertTriangle size={16} />{error}</div>}

      {loading && models.length === 0 ? (
        <div className="aurelius-card text-center py-12 text-[#9e9eb0]"><Loader2 size={32} className="mx-auto mb-3 animate-spin opacity-60" /><p>Loading models...</p></div>
      ) : models.length === 0 ? (
        <div className="aurelius-card text-center py-12 text-[#9e9eb0]"><Cpu size={32} className="mx-auto mb-3 opacity-40" /><p>No models found.</p></div>
      ) : (
        <div className="grid gap-4">
          {models.map((m: any) => (
            <div key={m.id} className="aurelius-card flex items-center justify-between">
              <div className="flex items-center gap-4">
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${stateColors[m.state] || stateColors.unloaded}`}>
                  {m.state === 'loaded' ? <Zap size={18} /> : <Cpu size={18} />}
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-[#e0e0e0]">{m.name}</h3>
                  <p className="text-xs text-[#9e9eb0]">{m.description}</p>
                  <div className="flex items-center gap-3 mt-1">
                    <span className="text-[10px] text-[#9e9eb0]/60">{fmtParams(m.parameterCount)} params</span>
                    <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full border ${stateColors[m.state] || stateColors.unloaded}`}>{m.state}</span>
                    {m.loadedAt && <span className="text-[10px] text-[#9e9eb0]/60">Loaded: {new Date(m.loadedAt).toLocaleTimeString()}</span>}
                  </div>
                </div>
              </div>
              <button
                onClick={() => toggleModel(m.id, m.state)}
                disabled={m.state === 'loading'}
                className={`aurelius-btn-outline flex items-center gap-1.5 text-sm disabled:opacity-50 ${m.state === 'loaded' ? 'text-rose-400 border-rose-500/20 hover:bg-rose-500/10' : 'text-emerald-400 border-emerald-500/20 hover:bg-emerald-500/10'}`}
              >
                {m.state === 'loading' ? <Loader2 size={14} className="animate-spin" /> : m.state === 'loaded' ? <Square size={14} /> : <Play size={14} />}
                {m.state === 'loaded' ? 'Unload' : m.state === 'loading' ? 'Loading...' : 'Load'}
              </button>
            </div>
          ))}
        </div>
      )}

      <div className="aurelius-card text-xs text-[#9e9eb0] flex items-start gap-2">
        <Info size={14} className="shrink-0 mt-0.5" />
        <p>Model states are simulated. To run actual inference, start the Python model server separately on port 8080 and the gateway will proxy requests.</p>
      </div>
    </div>
  )
}
