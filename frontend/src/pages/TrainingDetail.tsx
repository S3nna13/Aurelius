import { useState, useEffect, useMemo } from 'react'
import { useParams, Link } from 'react-router-dom'
import { ArrowLeft, Activity, Clock, Target, TrendingDown, Zap, Loader2, AlertTriangle } from 'lucide-react'
import { api } from '../api/AureliusClient'
import type { TrainingRunDetail } from '../api/types'

const statusColors: Record<string, string> = {
  running: 'text-emerald-400', completed: 'text-[#4fc3f7]', queued: 'text-amber-400', failed: 'text-rose-400',
}

function LineChartSVG({ data, color, height = 180 }: { data: number[]; color: string; height?: number }) {
  if (!data.length) return null
  const w = 600
  const max = Math.max(...data); const min = Math.min(...data)
  const range = max - min || 1
  const pad = 25
  const cw = w - pad * 2; const ch = height - pad * 2

  const pts = data.map((v, i) =>
    `${(i / (data.length - 1)) * cw + pad},${ch - ((v - min) / range) * ch + pad}`
  ).join(' ')

  const yTicks = 5
  const yLabels = Array.from({ length: yTicks + 1 }, (_, i) => min + (range * i) / yTicks)

  return (
    <svg viewBox={`0 0 ${w} ${height}`} className="w-full h-auto" preserveAspectRatio="xMidYMid meet">
      {yLabels.map((v, i) => (
        <text key={i} x={pad - 4} y={height - (i / yTicks) * ch - pad + 4} textAnchor="end" fontSize="10" fill="#9e9eb0">
          {v.toFixed(2)}
        </text>
      ))}
      {yLabels.map((_, i) => (
        <line key={i} x1={pad} y1={height - (i / yTicks) * ch - pad} x2={w - pad} y2={height - (i / yTicks) * ch - pad}
          stroke="#2d2d44" strokeWidth="0.5" />
      ))}
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.5" vectorEffect="non-scaling-stroke" />
    </svg>
  )
}

function Section({ title, icon: Icon, children }: { title: string; icon: typeof Activity; children: React.ReactNode }) {
  return (
    <div className="aurelius-card space-y-3">
      <h3 className="text-sm font-semibold text-[#e0e0e0] uppercase tracking-wider flex items-center gap-2">
        <Icon size={16} className="text-[#4fc3f7]" /> {title}
      </h3>
      {children}
    </div>
  )
}

function Metric({ label, value, color }: { label: string; value: string; color?: string }) {
  return (
    <div className="bg-[#0f0f1a] border border-[#2d2d44] rounded-lg p-3">
      <p className="text-[10px] text-[#9e9eb0] uppercase tracking-wider">{label}</p>
      <p className={`text-sm font-bold mt-0.5 ${color || 'text-[#e0e0e0]'}`}>{value}</p>
    </div>
  )
}

export default function TrainingDetail() {
  const { id } = useParams<{ id: string }>()
  const [run, setRun] = useState<TrainingRunDetail | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!id) return
    const fetch = async () => {
      try {
        setLoading(true)
        const data = await api.getTrainingRun(id)
        setRun(data)
      } catch (err) {
        setError(err instanceof Error ? err.message : 'Failed to load')
      } finally {
        setLoading(false)
      }
    }
    fetch()
  }, [id])

  const avgLoss = useMemo(() => {
    if (!run?.valLosses?.length) return 0
    return run.valLosses.reduce((a, b) => a + b, 0) / run.valLosses.length
  }, [run])

  if (loading) return (
    <div className="aurelius-card text-center py-16 text-[#9e9eb0]">
      <Loader2 size={32} className="mx-auto mb-3 animate-spin opacity-60" /><p>Loading run...</p>
    </div>
  )

  if (error || !run) return (
    <div className="aurelius-card border-rose-500/30 bg-rose-500/5 text-rose-300 text-sm flex items-center gap-2 p-4">
      <AlertTriangle size={16} />{error || 'Run not found'}
    </div>
  )

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-center gap-3">
        <Link to="/training" className="p-2 rounded-lg text-[#9e9eb0] hover:text-[#e0e0e0] hover:bg-[#2d2d44]/40 transition-colors">
          <ArrowLeft size={18} />
        </Link>
        <div>
          <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
            <Activity size={20} className="text-[#4fc3f7]" /> {run.name}
          </h2>
          <p className="text-xs text-[#9e9eb0] flex items-center gap-2 mt-0.5">
            <span className={`text-xs font-bold ${statusColors[run.status] || 'text-[#9e9eb0]'}`}>{run.status}</span>
            <span>{run.modelId}</span>
            <span>Epoch {run.currentEpoch}/{run.totalEpochs}</span>
          </p>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <Metric label="Best Val Loss" value={run.bestValLoss.toFixed(4)} color="text-emerald-400" />
        <Metric label="Current LR" value={run.currentLr > 0 ? run.currentLr.toExponential(2) : '—'} color="text-[#4fc3f7]" />
        <Metric label="Total Steps" value={String(run.totalSteps)} />
        <Metric label="Avg Val Loss" value={avgLoss.toFixed(4)} />
      </div>

      {run.valLosses.length > 0 && (
        <Section title="Training Loss" icon={TrendingDown}>
          <div className="flex items-center gap-4 text-xs text-[#9e9eb0] mb-2">
            <span className="flex items-center gap-1"><svg width="12" height="3"><line x1="0" y1="1.5" x2="12" y2="1.5" stroke="#4fc3f7" strokeWidth="2" /></svg> Train Loss</span>
            <span className="flex items-center gap-1"><svg width="12" height="3"><line x1="0" y1="1.5" x2="12" y2="1.5" stroke="#34d399" strokeWidth="2" /></svg> Val Loss</span>
          </div>
          <LineChartSVG data={run.trainLosses} color="#4fc3f7" />
          <LineChartSVG data={run.valLosses} color="#34d399" />
        </Section>
      )}

      {run.learningRates.length > 0 && (
        <Section title="Learning Rate Schedule" icon={Zap}>
          <LineChartSVG data={run.learningRates} color="#fbbf24" />
        </Section>
      )}

      {run.accuracies.length > 0 && (
        <Section title="Accuracy" icon={Target}>
          <LineChartSVG data={run.accuracies} color="#a78bfa" />
        </Section>
      )}

      <div className="aurelius-card text-xs text-[#9e9eb0] flex items-start gap-2">
        <Clock size={14} className="shrink-0 mt-0.5" />
        <p>Started: {run.startedAt > 0 ? new Date(run.startedAt * 1000).toLocaleString() : 'Not started'}</p>
      </div>
    </div>
  )
}
