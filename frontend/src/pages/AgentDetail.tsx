import { useState, useEffect, useCallback } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Bot, ArrowLeft, Activity,  Server, Zap, Loader2, Play, Pause, RefreshCw, Terminal, AlertTriangle } from 'lucide-react'
import { useToast } from '../components/ToastProvider'

interface AgentData {
  id: string
  state: string
  role: string
  metrics_json: string
}

interface ActivityEntry {
  id: string
  command: string
  success: boolean
  output: string
  timestamp: number
}

function timeAgo(ts: number): string {
  const diff = Date.now() / 1000 - ts
  if (diff < 60) return 'Just now'
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

const roleLabels: Record<string, string> = {
  'notification-router': 'Notification Router',
  'task-orchestrator': 'Task Orchestrator',
  'memory-manager': 'Memory Manager',
  'security-warden': 'Security Warden',
  'code-analyst': 'Code Analyst',
}

const agentColors: Record<string, string> = {
  hermes: '#4fc3f7',
  openclaw: '#34d399',
  cerebrum: '#fbbf24',
  vigil: '#f87171',
  thoth: '#a78bfa',
}

export default function AgentDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const { toast } = useToast()
  const [agent, setAgent] = useState<AgentData | null>(null)
  const [activity, setActivity] = useState<ActivityEntry[]>([])
  const [loading, setLoading] = useState(true)
  const [command, setCommand] = useState('')
  const [executing, setExecuting] = useState(false)

  const fetchAgent = useCallback(async () => {
    if (!id) return
    try {
      const [agentRes, activityRes] = await Promise.all([
        fetch(`/api/agents/${id}`),
        fetch('/api/activity?limit=20'),
      ])
      if (agentRes.ok) setAgent(await agentRes.json())
      if (activityRes.ok) {
        const data = await activityRes.json()
        setActivity(data.entries || [])
      }
    } catch (err) {
      toast(err instanceof Error ? err.message : 'Failed to load agent', 'error')
    } finally {
      setLoading(false)
    }
  }, [id, toast])

  // eslint-disable-next-line react-hooks/set-state-in-effect
  useEffect(() => { fetchAgent() }, [fetchAgent])

  const sendCommand = async () => {
    if (!command.trim() || executing || !id) return
    setExecuting(true)
    try {
      const res = await fetch(`/api/agents/${id}/command`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: command.trim() }),
      })
      const data = await res.json()
      if (data.success) {
        toast(`Command executed: ${command.trim()}`, 'success')
        setCommand('')
        fetchAgent()
      } else {
        toast(data.error || 'Command failed', 'error')
      }
    } catch {
      toast('Failed to execute command', 'error')
    } finally {
      setExecuting(false)
    }
  }

  const setState = async (newState: string) => {
    if (!id) return
    try {
      const res = await fetch(`/api/agents/${id}/state`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ state: newState }),
      })
      const data = await res.json()
      if (data.success) {
        toast(`Agent state changed to ${newState}`, 'success')
        fetchAgent()
      }
    } catch {
      toast('Failed to change state', 'error')
    }
  }

  const color = id ? agentColors[id.toLowerCase()] || '#4fc3f7' : '#4fc3f7'
  const agentActivity = activity.filter((a) => a.command.includes(`agent.${id}`) || a.command.includes(id || ''))

  if (loading) {
    return <div className="aurelius-card text-center py-16 text-[#9e9eb0]"><Loader2 size={32} className="mx-auto mb-3 animate-spin opacity-60" /><p>Loading agent...</p></div>
  }

  if (!agent) {
    return (
      <div className="aurelius-card text-center py-16">
        <AlertTriangle size={32} className="mx-auto mb-3 text-rose-400" />
        <p className="text-[#e0e0e0] font-medium">Agent not found</p>
        <p className="text-sm text-[#9e9eb0] mt-1">The agent "{id}" does not exist.</p>
        <button onClick={() => navigate('/')} className="aurelius-btn-outline mt-4 text-sm">Back to Dashboard</button>
      </div>
    )
  }

  const metrics = (() => {
    try { return JSON.parse(agent.metrics_json) } catch { return {} }
  })()

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-center gap-4">
        <button onClick={() => navigate('/')} className="p-2 rounded-lg text-[#9e9eb0] hover:text-[#e0e0e0] hover:bg-[#2d2d44]/40 transition-colors"><ArrowLeft size={18} /></button>
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl flex items-center justify-center" style={{ backgroundColor: `${color}20`, color }}><Bot size={20} /></div>
          <div><h2 className="text-lg font-bold text-[#e0e0e0]">{agent.id.replace(/^\w/, (c) => c.toUpperCase())}</h2><p className="text-sm text-[#9e9eb0]">{roleLabels[agent.role] || agent.role}</p></div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-6">
          <div className="aurelius-card space-y-4">
            <h3 className="text-sm font-semibold text-[#e0e0e0] uppercase tracking-wider flex items-center gap-2"><Activity size={16} className="text-[#4fc3f7]" /> Agent Activity</h3>
            {agentActivity.length === 0 ? <p className="text-sm text-[#9e9eb0] text-center py-6">No activity recorded yet.</p> : (
              <div className="space-y-2 max-h-[300px] overflow-y-auto">
                {agentActivity.reverse().map((entry) => (
                  <div key={entry.id} className="flex items-start gap-3 px-3 py-2 rounded-lg bg-[#0f0f1a]/50 border border-[#2d2d44]/50">
                    <div className={`mt-1 w-2 h-2 rounded-full shrink-0 ${entry.success ? 'bg-emerald-400' : 'bg-rose-400'}`} />
                    <div className="flex-1 min-w-0"><p className="text-sm text-[#e0e0e0]">{entry.command}</p><p className="text-xs text-[#9e9eb0] truncate">{entry.output}</p></div>
                    <span className="text-[10px] text-[#9e9eb0]/60 shrink-0">{timeAgo(entry.timestamp)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="aurelius-card space-y-4">
            <h3 className="text-sm font-semibold text-[#e0e0e0] uppercase tracking-wider flex items-center gap-2"><Terminal size={16} className="text-[#4fc3f7]" /> Command</h3>
            <div className="flex gap-2">
              <input type="text" placeholder="Send command to agent..." value={command} onChange={(e) => setCommand(e.target.value)} onKeyDown={(e) => { if (e.key === 'Enter') sendCommand() }} disabled={executing}
                className="flex-1 bg-[#0f0f1a] border border-[#2d2d44] rounded-lg px-3 py-2 text-sm text-[#e0e0e0] placeholder:text-[#9e9eb0] focus:outline-none focus:border-[#4fc3f7] disabled:opacity-50" />
              <button onClick={sendCommand} disabled={executing || !command.trim()} className="aurelius-btn px-4 text-sm disabled:opacity-50">{executing ? <Loader2 size={14} className="animate-spin" /> : 'Send'}</button>
            </div>
          </div>
        </div>

        <div className="space-y-6">
          <div className="aurelius-card space-y-4">
            <h3 className="text-sm font-semibold text-[#e0e0e0] uppercase tracking-wider flex items-center gap-2"><Server size={16} className="text-[#4fc3f7]" /> Status</h3>
            <div className="space-y-3">
              <div className="flex justify-between items-center"><span className="text-sm text-[#9e9eb0]">State</span>
                <span className={`text-xs font-bold px-2 py-0.5 rounded-full border ${agent.state.toUpperCase() === 'ACTIVE' || agent.state.toUpperCase() === 'RUNNING' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : agent.state.toUpperCase() === 'IDLE' ? 'bg-amber-500/10 text-amber-400 border-amber-500/20' : 'bg-rose-500/10 text-rose-400 border-rose-500/20'}`}>{agent.state}</span>
              </div>
              <div className="flex justify-between items-center"><span className="text-sm text-[#9e9eb0]">Role</span><span className="text-sm text-[#e0e0e0]">{roleLabels[agent.role] || agent.role}</span></div>
              <div className="flex justify-between items-center"><span className="text-sm text-[#9e9eb0]">Agent ID</span><span className="text-sm text-[#e0e0e0] font-mono">{agent.id}</span></div>
              {Object.entries(metrics).slice(0, 3).map(([k, v]) => (
                <div key={k} className="flex justify-between items-center"><span className="text-sm text-[#9e9eb0] capitalize">{k.replace(/_/g, ' ')}</span><span className="text-sm text-[#e0e0e0]">{String(v)}</span></div>
              ))}
            </div>
          </div>

          <div className="aurelius-card space-y-4">
            <h3 className="text-sm font-semibold text-[#e0e0e0] uppercase tracking-wider flex items-center gap-2"><Zap size={16} className="text-[#4fc3f7]" /> Controls</h3>
            <div className="space-y-2">
              <button onClick={() => setState('active')} disabled={agent.state === 'active'} className="w-full aurelius-btn-outline flex items-center gap-2 text-sm disabled:opacity-50"><Play size={14} /> Activate</button>
              <button onClick={() => setState('idle')} disabled={agent.state === 'idle'} className="w-full aurelius-btn-outline flex items-center gap-2 text-sm disabled:opacity-50"><Pause size={14} /> Idle</button>
              <button onClick={fetchAgent} className="w-full aurelius-btn-outline flex items-center gap-2 text-sm"><RefreshCw size={14} /> Refresh</button>
            </div>
          </div>
        </div>
      </div>
    </div>
  )
}
