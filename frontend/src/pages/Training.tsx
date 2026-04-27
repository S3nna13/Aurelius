import { useState, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { LineChart, Activity, RefreshCw, Loader2, AlertTriangle, Play, CheckCircle, Clock, TrendingDown, Zap } from 'lucide-react'
import { api } from '../api/AureliusClient'
import { useWebSocket } from '../hooks/useWebSocket'
import type { TrainingRunSummary } from '../api/types'

const statusStyles: Record<string, string> = {
  running: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
  completed: 'text-[#4fc3f7] bg-[#4fc3f7]/10 border-[#4fc3f7]/20',
  queued: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
  failed: 'text-rose-400 bg-rose-500/10 border-rose-500/20',
}

const statusIcons: Record<string, typeof Play> = {
  running: Activity, completed: CheckCircle, queued: Clock, failed: AlertTriangle,
}

export default function Training() {
  const [runs, setRuns] = useState<TrainingRunSummary[]>([])
  const [stats, setStats] = useState<{ total_runs: number; total_steps: number; avg_val_loss: number | null; best_val_loss: number | null; running_count: number } | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)
  const { connected, on, off, subscribe } = useWebSocket()

  const fetch = useCallback(async () => {
    try {
      const [runsRes, statsRes] = await Promise.all([
        api.listTrainingRuns(),
        api.get<{ total_runs: number; total_steps: number; avg_val_loss: number | null; best_val_loss: number | null; running_count: number }>('/training/stats'),
      ])
      setRuns(runsRes.runs)
      setStats(statsRes as any)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetch() }, [fetch])

  useEffect(() => {
    if (!connected) return
    subscribe('training')
  }, [connected, subscribe])

  useEffect(() => {
    const handler = (data: unknown) => {
      const d = data as Record<string, unknown>
      if (d.type === 'created' || d.type === 'update') fetch()
    }
    on('training', handler)
    return () => off('training', handler)
  }, [on, off, fetch])

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
          <LineChart size={20} className="text-[#4fc3f7]" /> Training Monitor
          {connected && <span className="text-[10px] font-bold text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20 ml-2">LIVE</span>}
        </h2>
        <div className="flex items-center gap-3">
          <span className="text-xs text-[#9e9eb0]">{runs.filter(r => r.status === 'running').length} active · {runs.length} total</span>
          <button onClick={fetch} disabled={loading} className="aurelius-btn-outline flex items-center gap-1.5 text-sm disabled:opacity-50">
            {loading ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />} Refresh
          </button>
        </div>
      </div>

      {error && <div className="aurelius-card border-rose-500/30 bg-rose-500/5 text-rose-300 text-sm flex items-center gap-2"><AlertTriangle size={16} />{error}</div>}

      {stats && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <div className="aurelius-card text-center py-3">
            <Activity size={16} className="mx-auto mb-1 text-[#4fc3f7]" />
            <p className="text-xl font-bold text-[#e0e0e0]">{stats.total_runs}</p>
            <p className="text-[10px] text-[#9e9eb0] uppercase tracking-wider">Total Runs</p>
          </div>
          <div className="aurelius-card text-center py-3">
            <TrendingDown size={16} className="mx-auto mb-1 text-emerald-400" />
            <p className="text-xl font-bold text-emerald-400">{stats.avg_val_loss !== null ? stats.avg_val_loss.toFixed(3) : '—'}</p>
            <p className="text-[10px] text-[#9e9eb0] uppercase tracking-wider">Avg Val Loss</p>
          </div>
          <div className="aurelius-card text-center py-3">
            <Zap size={16} className="mx-auto mb-1 text-amber-400" />
            <p className="text-xl font-bold text-amber-400">{stats.best_val_loss !== null ? stats.best_val_loss.toFixed(4) : '—'}</p>
            <p className="text-[10px] text-[#9e9eb0] uppercase tracking-wider">Best Loss</p>
          </div>
          <div className="aurelius-card text-center py-3">
            <Clock size={16} className="mx-auto mb-1 text-[#9e9eb0]" />
            <p className="text-xl font-bold text-[#e0e0e0]">{stats.total_steps.toLocaleString()}</p>
            <p className="text-[10px] text-[#9e9eb0] uppercase tracking-wider">Total Steps</p>
          </div>
        </div>
      )}

      {loading && runs.length === 0 ? (
        <div className="aurelius-card text-center py-12 text-[#9e9eb0]"><Loader2 size={32} className="mx-auto mb-3 animate-spin opacity-60" /><p>Loading training runs...</p></div>
      ) : runs.length === 0 ? (
        <div className="aurelius-card text-center py-12 text-[#9e9eb0]"><LineChart size={32} className="mx-auto mb-3 opacity-40" /><p>No training runs yet.</p></div>
      ) : (
        <div className="grid gap-4">
          {runs.map((run) => {
            const StatusIcon = statusIcons[run.status] || Clock
            return (
              <Link key={run.id} to={`/training/${run.id}`} className="aurelius-card flex items-center gap-4 hover:border-[#4fc3f7]/30 transition-colors group">
                <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${statusStyles[run.status] || statusStyles.queued}`}>
                  <StatusIcon size={18} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2">
                    <h3 className="text-sm font-semibold text-[#e0e0e0]">{run.name}</h3>
                    <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full border ${statusStyles[run.status] || statusStyles.queued}`}>{run.status}</span>
                  </div>
                  <div className="flex items-center gap-4 mt-1 text-xs text-[#9e9eb0]">
                    <span>Epoch {run.currentEpoch}/{run.totalEpochs}</span>
                    <span>Best val loss: {run.bestValLoss.toFixed(3)}</span>
                    {run.currentLr > 0 && <span>LR: {run.currentLr.toExponential(1)}</span>}
                    <span>{run.totalSteps} steps</span>
                  </div>
                </div>
              </Link>
            )
          })}
        </div>
      )}
    </div>
  )
}
