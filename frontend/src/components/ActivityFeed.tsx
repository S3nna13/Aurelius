import { useState, useEffect } from 'react'
import { Activity, Clock, Loader2, CheckCircle, XCircle } from 'lucide-react'
import { useWebSocket } from '../hooks/useWebSocket'
import { EmptyState } from './EmptyState'

interface ActivityEntry {
  id: string
  command: string
  success: boolean
  output: string
  timestamp: number
}

interface ActivityFeedProps {
  limit?: number
  showHeader?: boolean
  compact?: boolean
}

export function ActivityFeed({ limit = 20, showHeader = true, compact = false }: ActivityFeedProps) {
  const [entries, setEntries] = useState<ActivityEntry[]>([])
  const [loading, setLoading] = useState(true)
  const ws = useWebSocket()

  useEffect(() => {
    const load = async () => {
      try {
        const res = await fetch('/api/activity')
        if (res.ok) {
          const data = await res.json()
          setEntries((data.entries || []).slice(0, limit))
        }
      } catch { /* ignore */ }
      setLoading(false)
    }
    load()
  }, [limit])

  useEffect(() => {
    ws.on('activity:new', (payload) => {
      const p = payload as { command: string; timestamp: number }
      setEntries((prev) => {
        const next = [{ id: `live-${Date.now()}`, command: p.command, success: true, output: '', timestamp: p.timestamp || Date.now() }, ...prev]
        return next.slice(0, limit)
      })
    })
  }, [ws, limit])

  if (loading) return <div className="flex items-center justify-center py-6"><Loader2 size={16} className="animate-spin text-[#9e9eb0]" /></div>

  return (
    <div className="space-y-3">
      {showHeader && (
        <h3 className="text-sm font-semibold text-[#e0e0e0] uppercase tracking-wider flex items-center gap-2">
          <Activity size={16} className="text-[#4fc3f7]" /> Activity
        </h3>
      )}
      {entries.length === 0 ? (
        <EmptyState icon={<Activity size={24} />} title="No activity yet" />
      ) : (
        <div className="space-y-1">
          {entries.map((entry) => (
            <div key={entry.id} className={`flex items-start gap-3 px-3 py-2 rounded-lg ${compact ? 'py-1.5' : 'py-2'} hover:bg-[#0f0f1a]/40 transition-colors`}>
              {entry.success ? <CheckCircle size={14} className="mt-0.5 shrink-0 text-emerald-400" /> : <XCircle size={14} className="mt-0.5 shrink-0 text-rose-400" />}
              <div className="flex-1 min-w-0">
                <p className={`text-[#e0e0e0] ${compact ? 'text-xs' : 'text-sm'} truncate`}>{entry.command}</p>
                {!compact && entry.output && <p className="text-xs text-[#9e9eb0] truncate mt-0.5">{entry.output}</p>}
              </div>
              <span className="text-[10px] text-[#9e9eb0]/40 shrink-0 flex items-center gap-1"><Clock size={10} />{timeAgo(entry.timestamp)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function timeAgo(ts: number): string {
  const diff = Date.now() / 1000 - ts
  if (diff < 60) return 'now'
  if (diff < 3600) return `${Math.floor(diff / 60)}m`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h`
  return `${Math.floor(diff / 86400)}d`
}
