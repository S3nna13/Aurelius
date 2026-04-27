import { useState, useEffect, useCallback } from 'react'
import { Database, RefreshCw, Loader2, Download, CheckCircle, XCircle } from 'lucide-react'
import { DataGrid, type DataGridColumn } from '../components/DataGrid'
import { Tabs } from '../components/ui/Tabs'
import { Badge } from '../components/ui/Badge'
import { Spinner } from '../components/Spinner'
import { Breadcrumbs } from '../components/Breadcrumbs'
import { downloadJSON } from '../utils/export'
import { useToast } from '../components/ToastProvider'

interface AgentRow { id: string; state: string; role: string }
interface ActivityRow { id: string; timestamp: number; command: string; success: boolean; output: string }
interface NotificationRow { id: string; title: string; priority: string; category: string; read: boolean; timestamp: number; channel: string }
interface LogRow { timestamp: string; level: string; logger: string; message: string }

function timeAgo(ts: number): string {
  const diff = Date.now() / 1000 - ts
  if (diff < 60) return 'now'
  if (diff < 3600) return `${Math.floor(diff / 60)}m`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h`
  return `${Math.floor(diff / 86400)}d`
}

// eslint-disable-next-line @typescript-eslint/no-explicit-any
const AnyDataGrid = DataGrid as any

export default function DataExplorer() {
  const { toast } = useToast()
  const [, setTab] = useState('agents')
  const [agents, setAgents] = useState<AgentRow[]>([])
  const [activity, setActivity] = useState<ActivityRow[]>([])
  const [notifications, setNotifications] = useState<NotificationRow[]>([])
  const [logs, setLogs] = useState<LogRow[]>([])
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const fetchAll = useCallback(async () => {
    setLoading(true)
    setError(null)
    try {
      const [agentsRes, activityRes, notifsRes, logsRes] = await Promise.all([
        fetch('/api/agents'),
        fetch('/api/activity?limit=200'),
        fetch('/api/notifications?limit=200'),
        fetch('/api/logs?limit=200'),
      ])
      if (agentsRes.ok) { const d = await agentsRes.json(); setAgents(d.agents || []) }
      if (activityRes.ok) { const d = await activityRes.json(); setActivity(d.entries || []) }
      if (notifsRes.ok) { const d = await notifsRes.json(); setNotifications(d.notifications || []) }
      if (logsRes.ok) { const d = await logsRes.json(); setLogs(d.logs || []) }
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load data')
    }
    setLoading(false)
  }, [])

  useEffect(() => { fetchAll() }, [fetchAll])

  const exportAll = () => {
    downloadJSON({ agents, activity, notifications, logs, exportedAt: new Date().toISOString() }, `aurelius-data-${Date.now()}.json`)
    toast('Data exported', 'success')
  }

  const agentCols: DataGridColumn<AgentRow>[] = [
    { key: 'id', header: 'ID', sortable: true },
    { key: 'state', header: 'State', sortable: true, render: (a) => <Badge variant={a.state === 'active' ? 'success' : a.state === 'idle' ? 'warning' : 'default'}>{a.state}</Badge> },
    { key: 'role', header: 'Role', sortable: true },
  ]

  const activityCols: DataGridColumn<ActivityRow>[] = [
    { key: 'timestamp', header: 'Time', sortable: true, render: (a) => <span className="text-[#9e9eb0] text-xs">{timeAgo(a.timestamp)}</span> },
    { key: 'command', header: 'Command', sortable: true },
    { key: 'success', header: 'Status', render: (a) => a.success ? <CheckCircle size={14} className="text-emerald-400" /> : <XCircle size={14} className="text-rose-400" /> },
    { key: 'output', header: 'Output' },
  ]

  const notifCols: DataGridColumn<NotificationRow>[] = [
    { key: 'timestamp', header: 'Time', sortable: true, render: (n) => <span className="text-[#9e9eb0] text-xs">{timeAgo(n.timestamp)}</span> },
    { key: 'priority', header: 'Priority', render: (n) => <Badge variant={n.priority === 'high' ? 'error' : n.priority === 'medium' ? 'warning' : 'info'}>{n.priority}</Badge> },
    { key: 'title', header: 'Title', sortable: true },
    { key: 'category', header: 'Category', render: (n) => <Badge>{n.category}</Badge> },
    { key: 'read', header: 'Read', render: (n) => n.read ? <CheckCircle size={14} className="text-emerald-400" /> : <XCircle size={14} className="text-rose-400" /> },
  ]

  const logCols: DataGridColumn<LogRow>[] = [
    { key: 'timestamp', header: 'Timestamp', sortable: true },
    { key: 'level', header: 'Level', render: (l) => <Badge variant={l.level === 'error' ? 'error' : l.level === 'warn' ? 'warning' : 'info'}>{l.level}</Badge> },
    { key: 'logger', header: 'Logger', sortable: true },
    { key: 'message', header: 'Message' },
  ]

  const tabs = [
    { id: 'agents', label: 'Agents', badge: agents.length },
    { id: 'activity', label: 'Activity', badge: activity.length },
    { id: 'notifications', label: 'Notifications', badge: notifications.length },
    { id: 'logs', label: 'Logs', badge: logs.length },
  ]

  return (
    <div className="space-y-6">
      <Breadcrumbs items={[{ label: 'Data Explorer' }]} />
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2"><Database size={20} className="text-[#4fc3f7]" /> Data Explorer</h2>
        <div className="flex gap-2">
          <button onClick={exportAll} disabled={loading} className="aurelius-btn-outline flex items-center gap-1.5 text-sm disabled:opacity-50"><Download size={14} /> Export</button>
          <button onClick={fetchAll} disabled={loading} className="aurelius-btn-outline flex items-center gap-1.5 text-sm disabled:opacity-50">{loading ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />} Refresh</button>
        </div>
      </div>

      {error && <div className="aurelius-card border-rose-500/30 bg-rose-500/5 text-rose-300 p-3 text-sm">{error}</div>}

      <Tabs tabs={tabs} defaultTab="agents" onChange={setTab}>
        {(active) => {
          if (loading) return <Spinner text="Loading data..." />
          switch (active) {
            case 'agents':
              return <AnyDataGrid columns={agentCols} data={agents} keyField="id" searchable />
            case 'activity':
              return <AnyDataGrid columns={activityCols} data={activity} keyField="id" searchable />
            case 'notifications':
              return <AnyDataGrid columns={notifCols} data={notifications} keyField="id" searchable />
            case 'logs':
              return <AnyDataGrid columns={logCols} data={logs} keyField={(l: any) => `${l.timestamp}-${l.message}`} searchable />
            default:
              return null
          }
        }}
      </Tabs>
    </div>
  )
}
