import { useState, useEffect } from 'react'
import { GitBranch, Loader2, RefreshCw, Play, Square, Clock, CheckCircle, XCircle, AlertTriangle } from 'lucide-react'
import { useToast } from '../components/ToastProvider'

interface Workflow {
  id: string
  name: string
  status: 'running' | 'completed' | 'failed' | 'pending'
  steps: number
  currentStep: number
  startedAt: string
  duration?: string
}

const mockWorkflows: Workflow[] = [
  { id: 'wf-1', name: 'Data Pipeline', status: 'running', steps: 5, currentStep: 3, startedAt: new Date(Date.now() - 120000).toISOString(), duration: '2m 34s' },
  { id: 'wf-2', name: 'Model Evaluation', status: 'completed', steps: 8, currentStep: 8, startedAt: new Date(Date.now() - 3600000).toISOString(), duration: '45m 12s' },
  { id: 'wf-3', name: 'Training Run', status: 'completed', steps: 3, currentStep: 3, startedAt: new Date(Date.now() - 7200000).toISOString(), duration: '1h 12m' },
  { id: 'wf-4', name: 'Data Export', status: 'failed', steps: 4, currentStep: 2, startedAt: new Date(Date.now() - 600000).toISOString(), duration: '3m 45s' },
  { id: 'wf-5', name: 'System Backup', status: 'pending', steps: 6, currentStep: 0, startedAt: new Date().toISOString(), duration: '-' },
]

const statusColors: Record<string, string> = {
  running: 'text-[#4fc3f7] bg-[#4fc3f7]/10 border-[#4fc3f7]/20',
  completed: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
  failed: 'text-rose-400 bg-rose-500/10 border-rose-500/20',
  pending: 'text-[#9e9eb0] bg-[#2d2d44]/20 border-[#2d2d44]/40',
}

const statusIcons: Record<string, typeof Play> = {
  running: Play, completed: CheckCircle, failed: XCircle, pending: AlertTriangle,
}

export default function Workflows() {
  const { toast } = useToast()
  const [workflows, setWorkflows] = useState<Workflow[]>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    const load = async () => {
      setLoading(true)
      try {
        const res = await fetch('/api/status')
        if (res.ok) {
          const data = await res.json()
          if (data.workflows) setWorkflows(data.workflows)
          else setWorkflows(mockWorkflows)
        } else {
          setWorkflows(mockWorkflows)
        }
      } catch {
        setWorkflows(mockWorkflows)
      }
      setLoading(false)
    }
    load()
  }, [])

  const startWorkflow = async (id: string) => {
    toast(`Workflow ${id} triggered`, 'success')
    setWorkflows((prev) => prev.map((w) => w.id === id ? { ...w, status: 'running', currentStep: 1, startedAt: new Date().toISOString() } : w))
  }

  const stopWorkflow = async (id: string) => {
    toast(`Workflow ${id} stopped`, 'info')
    setWorkflows((prev) => prev.map((w) => w.id === id ? { ...w, status: 'pending', currentStep: 0 } : w))
  }

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2"><GitBranch size={20} className="text-[#4fc3f7]" /> Workflows</h2>
        <button onClick={() => setLoading(true)} disabled={loading} className="aurelius-btn-outline flex items-center gap-1.5 text-sm disabled:opacity-50"><RefreshCw size={14} /> Refresh</button>
      </div>

      {loading ? (
        <div className="aurelius-card text-center py-12 text-[#9e9eb0]"><Loader2 size={24} className="mx-auto mb-2 animate-spin opacity-60" /><p>Loading workflows...</p></div>
      ) : (
        <div className="space-y-3">
          {workflows.map((wf) => {
            const StatusIcon = statusIcons[wf.status] || GitBranch
            return (
              <div key={wf.id} className="aurelius-card space-y-4">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <StatusIcon size={18} className={wf.status === 'running' ? 'text-[#4fc3f7]' : wf.status === 'completed' ? 'text-emerald-400' : wf.status === 'failed' ? 'text-rose-400' : 'text-[#9e9eb0]'} />
                    <div>
                      <p className="text-sm font-medium text-[#e0e0e0]">{wf.name}</p>
                      <p className="text-[10px] text-[#9e9eb0]">{wf.id}</p>
                    </div>
                  </div>
                  <div className="flex items-center gap-2">
                    <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${statusColors[wf.status]}`}>{wf.status}</span>
                    {wf.status === 'running' ? (
                      <button onClick={() => stopWorkflow(wf.id)} className="p-1.5 rounded text-[#9e9eb0] hover:text-rose-400 hover:bg-rose-500/10 transition-colors"><Square size={14} /></button>
                    ) : (
                      <button onClick={() => startWorkflow(wf.id)} className="p-1.5 rounded text-[#9e9eb0] hover:text-emerald-400 hover:bg-emerald-500/10 transition-colors"><Play size={14} /></button>
                    )}
                  </div>
                </div>
                <div className="space-y-2">
                  <div className="flex items-center justify-between text-xs text-[#9e9eb0]">
                    <span>Progress: {wf.currentStep}/{wf.steps} steps</span>
                    <span className="flex items-center gap-1"><Clock size={10} />{wf.duration || 'In progress'}</span>
                  </div>
                  <div className="w-full bg-[#0f0f1a] rounded-full h-1.5">
                    <div className={`h-1.5 rounded-full transition-all duration-500 ${wf.status === 'failed' ? 'bg-rose-400' : wf.status === 'completed' ? 'bg-emerald-400' : wf.status === 'running' ? 'bg-[#4fc3f7] animate-pulse' : 'bg-[#2d2d44]'}`}
                      style={{ width: `${wf.steps > 0 ? (wf.currentStep / wf.steps) * 100 : 0}%` }} />
                  </div>
                </div>
              </div>
            )
          })}
        </div>
      )}
    </div>
  )
}
