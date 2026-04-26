import { useState, useEffect, useCallback } from 'react'
import { Link } from 'react-router-dom'
import { LineChart, Activity, RefreshCw, Loader2, AlertTriangle, Play, CheckCircle, Clock } from 'lucide-react'
import { api } from '../api/AureliusClient'
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

function MiniSparkline({ data, color }: { data: number[]; color: string }) {
  if (!data.length) return null
  const max = Math.max(...data)
  const min = Math.min(...data)
  const range = max - min || 1
  const w = 120; const h = 32
  const pts = data.map((v, i) => `${(i / (data.length - 1)) * w},${h - ((v - min) / range) * h}`).join(' ')
  return (
    <svg width={w} height={h} className="shrink-0">
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
    </svg>
  )
}

export default function Training() {
  const [runs, setRuns] = useState<TrainingRunSummary[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetch = useCallback(async () => {
    try {
      const res = await api.listTrainingRuns()
      setRuns(res.runs)
      setError(null)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load')
    } finally {
      setLoading(false)
    }
  }, [])

  useEffect(() => { fetch() }, [fetch])

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
          <LineChart size={20} className="text-[#4fc3f7]" /> Training Monitor
        </h2>
        <div className="flex items-center gap-3">
          <span className="text-xs text-[#9e9eb0]">{runs.filter(r => r.status === 'running').length} active</span>
          <button onClick={fetch} disabled={loading} className="aurelius-btn-outline flex items-center gap-1.5 text-sm disabled:opacity-50">
            {loading ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />} Refresh
          </button>
        </div>
      </div>

      {error && <div className="aurelius-card border-rose-500/30 bg-rose-500/5 text-rose-300 text-sm flex items-center gap-2"><AlertTriangle size={16} />{error}</div>}

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
                <MiniSparkline data={[]} color="#4fc3f7" />
                <span className="text-[10px] text-[#9eeb0] opacity-0 group-hover:opacity-100 transition-opacity">&rarr;</span>
              </Link>
            )
          })}
        </div>
      )}
    </div>
  )
}
