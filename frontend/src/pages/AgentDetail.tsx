import { useState, useEffect } from 'react'
import { useParams, useNavigate } from 'react-router-dom'
import { Bot, ArrowLeft, Activity, Server, Zap, Loader2, Play, Pause, AlertTriangle } from 'lucide-react'
import { useAgentStore } from '../stores/agentStore'
import { api } from '../api/AureliusClient'

function timeAgo(ts: number): string {
  const diff = Date.now() / 1000 - ts
  if (diff < 60) return 'Just now'
  if (diff < 3600) return `${Math.floor(diff / 60)}m ago`
  if (diff < 86400) return `${Math.floor(diff / 3600)}h ago`
  return `${Math.floor(diff / 86400)}d ago`
}

const agentColors: Record<string, string> = {
  hermes: '#4fc3f7', openclaw: '#34d399', cerebrum: '#fbbf24', vigil: '#f87171',
}

export default function AgentDetail() {
  const { id } = useParams<{ id: string }>()
  const navigate = useNavigate()
  const agents = useAgentStore((s) => s.agents)
  const updateAgentState = useAgentStore((s) => s.updateAgentState)
  const [activity, setActivity] = useState<{ id: string; command: string; success: boolean; output: string; timestamp: number }[]>([])
  const [loading, setLoading] = useState(true)

  const agent = agents.find((a) => a.id === id)

  useEffect(() => {
    if (!id) return
    api.getActivity(50).then((res) => {
      setActivity(res.entries || [])
      setLoading(false)
    }).catch(() => setLoading(false))
  }, [id])

  if (!id) return null
  if (!agent) return (
    <div className="aurelius-card border-rose-500/30 bg-rose-500/5 text-rose-300 text-sm flex items-center gap-2 p-4">
      <AlertTriangle size={16} /> Agent not found
    </div>
  )

  const isActive = ['ACTIVE', 'RUNNING'].includes(agent.state.toUpperCase())
  const isIdle = agent.state.toUpperCase() === 'IDLE'
  const color = agentColors[agent.id] || '#4fc3f7'

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <button onClick={() => navigate(-1)} className="p-2 rounded-lg text-[#9e9eb0] hover:text-[#e0e0e0] hover:bg-[#2d2d44]/40 transition-colors">
          <ArrowLeft size={18} />
        </button>
        <div className="w-10 h-10 rounded-full flex items-center justify-center" style={{ backgroundColor: `${color}20`, color }}>
          <Bot size={20} />
        </div>
        <div>
          <h2 className="text-lg font-bold text-[#e0e0e0]">{agent.id.replace(/^\w/, (c) => c.toUpperCase())}</h2>
          <p className="text-xs text-[#9e9eb0]">{agent.role}</p>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="aurelius-card space-y-2">
          <div className="flex items-center gap-2 text-xs text-[#9e9eb0] uppercase tracking-wider">
            {isActive ? <Play size={14} className="text-emerald-400" /> : isIdle ? <Pause size={14} className="text-amber-400" /> : <Zap size={14} className="text-rose-400" />}
            State
          </div>
          <p className={`text-2xl font-bold ${isActive ? 'text-emerald-400' : isIdle ? 'text-amber-400' : 'text-rose-400'}`}>{agent.state}</p>
        </div>
        <div className="aurelius-card space-y-2">
          <div className="flex items-center gap-2 text-xs text-[#9e9eb0] uppercase tracking-wider"><Server size={14} className="text-[#4fc3f7]" /> Role</div>
          <p className="text-2xl font-bold text-[#e0e0e0]">{agent.role}</p>
        </div>
        <div className="aurelius-card space-y-2">
          <div className="flex items-center gap-2 text-xs text-[#9e9eb0] uppercase tracking-wider"><Activity size={14} className="text-[#4fc3f7]" /> Activity</div>
          <p className="text-2xl font-bold text-[#e0e0e0]">{activity.length}</p>
        </div>
      </div>

      <div className="aurelius-card space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-[#e0e0e0] uppercase tracking-wider flex items-center gap-2"><Activity size={14} className="text-[#4fc3f7]" /> Recent Activity</h3>
          <button onClick={() => updateAgentState(agent.id, isActive ? 'IDLE' : 'ACTIVE')}
            className={`aurelius-btn-outline flex items-center gap-1.5 text-xs ${isActive ? 'text-amber-400 border-amber-500/20 hover:bg-amber-500/10' : 'text-emerald-400 border-emerald-500/20 hover:bg-emerald-500/10'}`}>
            {isActive ? <Pause size={12} /> : <Play size={12} />}
            {isActive ? 'Pause' : 'Activate'}
          </button>
        </div>
        <div className="space-y-2 max-h-[400px] overflow-y-auto">
          {loading ? (
            <div className="text-center py-8 text-[#9e9eb0]"><Loader2 size={20} className="mx-auto mb-2 animate-spin opacity-60" /></div>
          ) : activity.length === 0 ? (
            <p className="text-sm text-[#9e9eb0] text-center py-6">No recent activity.</p>
          ) : (
            activity.map((entry) => (
              <div key={entry.id} className="flex items-start gap-3 px-3 py-2 rounded-lg bg-[#0f0f1a]/50 border border-[#2d2d44]/50">
                <div className={`mt-1 w-2 h-2 rounded-full shrink-0 ${entry.success ? 'bg-emerald-400' : 'bg-rose-400'}`} />
                <div className="flex-1 min-w-0">
                  <p className="text-sm text-[#e0e0e0] truncate">{entry.command}</p>
                  <p className="text-xs text-[#9e9eb0] truncate">{entry.output}</p>
                  <p className="text-[10px] text-[#9e9eb0]/60 mt-0.5">{timeAgo(entry.timestamp)}</p>
                </div>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  )
}
