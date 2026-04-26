import { useState, useCallback } from 'react';
import {
  GitBranch,
  Play,
  Clock,
  CheckCircle2,
  XCircle,
  RotateCcw,
  RefreshCw,
  Search,
  Loader2,
  X,
  AlertTriangle,
} from 'lucide-react';
import { useApi } from '../hooks/useApi';
import { useToast } from '../components/ToastProvider';

interface Workflow {
  id: string;
  name: string;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled' | 'idle';
  last_run: number;
  duration: number;
  event_count: number;
}

interface WorkflowEvent {
  type: string;
  message: string;
  timestamp: number;
}

interface WorkflowDetail extends Workflow {
  events: WorkflowEvent[];
}

function timeAgo(ts: number): string {
  const diff = Date.now() / 1000 - ts;
  if (diff < 60) return 'Just now';
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} hr ago`;
  return `${Math.floor(diff / 86400)} days ago`;
}

function formatDuration(sec: number): string {
  if (sec < 1) return `${(sec * 1000).toFixed(0)}ms`;
  if (sec < 60) return `${sec.toFixed(1)}s`;
  const m = Math.floor(sec / 60);
  const s = sec % 60;
  return `${m}m ${s.toFixed(0)}s`;
}

const statusConfig: Record<string, { icon: typeof CheckCircle2; classes: string; label: string }> = {
  running: { icon: RotateCcw, classes: 'bg-[#4fc3f7]/10 text-[#4fc3f7] border-[#4fc3f7]/20', label: 'Running' },
  completed: { icon: CheckCircle2, classes: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20', label: 'Completed' },
  failed: { icon: XCircle, classes: 'bg-rose-500/10 text-rose-400 border-rose-500/20', label: 'Failed' },
  pending: { icon: Clock, classes: 'bg-[#2d2d44]/20 text-[#9e9eb0] border-[#2d2d44]/40', label: 'Pending' },
  cancelled: { icon: X, classes: 'bg-amber-500/10 text-amber-400 border-amber-500/20', label: 'Cancelled' },
  idle: { icon: Clock, classes: 'bg-[#2d2d44]/20 text-[#9e9eb0] border-[#2d2d44]/40', label: 'Idle' },
};

function StatusBadge({ status }: { status: string }) {
  const cfg = statusConfig[status] || statusConfig.idle;
  const Icon = cfg.icon;
  return (
    <span className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium border ${cfg.classes}`}>
      <Icon size={12} className={status === 'running' ? 'animate-spin' : ''} />
      {cfg.label}
    </span>
  );
}

export default function Workflows() {
  const [search, setSearch] = useState('');
  const [selected, setSelected] = useState<WorkflowDetail | null>(null);
  const [triggering, setTriggering] = useState<string | null>(null);
  const { toast } = useToast();

  const {
    data: wfData,
    loading,
    error,
    refresh: refreshWorkflows,
  } = useApi<{ workflows: Workflow[]; summary: { total: number; running: number; completed: number; failed: number } }>('/workflows', {
    refreshInterval: 5000,
    retries: 2,
    timeout: 8000,
  });

  const workflows = wfData?.workflows || [];
  const summary = wfData?.summary || { total: 0, running: 0, completed: 0, failed: 0 };

  const openDetail = useCallback(async (wfId: string) => {
    try {
      const res = await fetch(`/api/workflows/${wfId}`, {
        signal: AbortSignal.timeout(8000),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setSelected(data as WorkflowDetail);
    } catch (err) {
      toast(err instanceof Error ? err.message : 'Failed to load workflow details', 'error');
    }
  }, [toast]);

  const triggerWorkflow = useCallback(async (wfId: string, trigger: string) => {
    setTriggering(wfId);
    try {
      const res = await fetch(`/api/workflows/${wfId}/trigger`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ trigger }),
        signal: AbortSignal.timeout(15000),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      await refreshWorkflows();
      if (selected && selected.id === wfId) {
        await openDetail(wfId);
      }
      toast(`Workflow ${trigger} triggered`, 'success');
    } catch (err) {
      toast(err instanceof Error ? err.message : 'Trigger failed', 'error');
    } finally {
      setTriggering(null);
    }
  }, [refreshWorkflows, selected, openDetail, toast]);

  const filtered = workflows.filter(
    (w) =>
      w.name.toLowerCase().includes(search.toLowerCase()) ||
      w.id.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
          <GitBranch size={20} className="text-[#4fc3f7]" />
          Workflows
        </h2>
        <div className="relative">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#9e9eb0]" />
          <input
            type="text"
            placeholder="Search workflows..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="bg-[#0f0f1a] border border-[#2d2d44] rounded-lg pl-9 pr-4 py-2 text-sm text-[#e0e0e0] placeholder:text-[#9e9eb0] focus:outline-none focus:border-[#4fc3f7] w-full sm:w-64"
          />
        </div>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: 'Total', value: summary.total, color: 'text-[#e0e0e0]' },
          { label: 'Running', value: summary.running, color: 'text-[#4fc3f7]' },
          { label: 'Completed', value: summary.completed, color: 'text-emerald-400' },
          { label: 'Failed', value: summary.failed, color: 'text-rose-400' },
        ].map((s) => (
          <div key={s.label} className="aurelius-card text-center py-3">
            <p className={`text-2xl font-bold ${s.color}`}>{s.value}</p>
            <p className="text-xs text-[#9e9eb0] uppercase tracking-wider mt-1">{s.label}</p>
          </div>
        ))}
      </div>

      {loading && workflows.length === 0 && (
        <div className="aurelius-card text-center py-12 text-[#9e9eb0]">
          <Loader2 size={32} className="mx-auto mb-3 animate-spin opacity-60" />
          <p>Loading workflows...</p>
        </div>
      )}

      {error && (
        <div className="aurelius-card border-rose-500/30 bg-rose-500/5 text-rose-300">
          <AlertTriangle size={18} className="inline mr-2" />
          {error.message}
          <button
            onClick={refreshWorkflows}
            className="ml-4 text-xs underline hover:text-rose-200"
          >
            Retry
          </button>
        </div>
      )}

      <div className="space-y-2">
        {filtered.map((wf) => (
          <div
            key={wf.id}
            className="aurelius-card flex flex-col sm:flex-row sm:items-center justify-between gap-3 hover:border-[#4fc3f7]/30 transition-colors group"
          >
            <div
              className="flex items-center gap-4 flex-1 cursor-pointer"
              onClick={() => openDetail(wf.id)}
            >
              <div className="w-9 h-9 rounded-lg bg-[#4fc3f7]/10 text-[#4fc3f7] flex items-center justify-center border border-[#4fc3f7]/20 shrink-0">
                <GitBranch size={18} />
              </div>
              <div className="min-w-0">
                <h3 className="text-sm font-semibold text-[#e0e0e0] group-hover:text-[#4fc3f7] transition-colors truncate">
                  {wf.name}
                </h3>
                <p className="text-xs text-[#9e9eb0]">{wf.id}</p>
              </div>
            </div>
            <div className="flex items-center gap-3 sm:gap-4">
              <div className="text-right hidden sm:block">
                <p className="text-[10px] text-[#9e9eb0] uppercase tracking-wider">Last run</p>
                <p className="text-sm text-[#e0e0e0]">{timeAgo(wf.last_run)}</p>
              </div>
              <div className="text-right hidden sm:block">
                <p className="text-[10px] text-[#9e9eb0] uppercase tracking-wider">Duration</p>
                <p className="text-sm text-[#e0e0e0]">{formatDuration(wf.duration)}</p>
              </div>
              <StatusBadge status={wf.status} />
              <div className="flex gap-1">
                {wf.status === 'idle' || wf.status === 'pending' || wf.status === 'failed' || wf.status === 'completed' || wf.status === 'cancelled' ? (
                  <button
                    onClick={() => triggerWorkflow(wf.id, 'start')}
                    disabled={triggering === wf.id}
                    className="w-8 h-8 rounded-lg bg-emerald-500/10 text-emerald-400 border border-emerald-500/20 flex items-center justify-center hover:bg-emerald-500/20 transition-colors disabled:opacity-50"
                    title="Start"
                  >
                    {triggering === wf.id ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
                  </button>
                ) : (
                  <button
                    onClick={() => triggerWorkflow(wf.id, 'complete')}
                    disabled={triggering === wf.id}
                    className="w-8 h-8 rounded-lg bg-[#4fc3f7]/10 text-[#4fc3f7] border border-[#4fc3f7]/20 flex items-center justify-center hover:bg-[#4fc3f7]/20 transition-colors disabled:opacity-50"
                    title="Complete"
                  >
                    {triggering === wf.id ? <Loader2 size={14} className="animate-spin" /> : <CheckCircle2 size={14} />}
                  </button>
                )}
                <button
                  onClick={() => triggerWorkflow(wf.id, 'fail')}
                  disabled={triggering === wf.id}
                  className="w-8 h-8 rounded-lg bg-rose-500/10 text-rose-400 border border-rose-500/20 flex items-center justify-center hover:bg-rose-500/20 transition-colors disabled:opacity-50"
                  title="Fail"
                >
                  <XCircle size={14} />
                </button>
                <button
                  onClick={() => triggerWorkflow(wf.id, 'reset')}
                  disabled={triggering === wf.id}
                  className="w-8 h-8 rounded-lg bg-[#2d2d44]/30 text-[#9e9eb0] border border-[#2d2d44]/40 flex items-center justify-center hover:bg-[#2d2d44]/50 transition-colors disabled:opacity-50"
                  title="Reset"
                >
                  <RefreshCw size={14} />
                </button>
              </div>
            </div>
          </div>
        ))}
      </div>

      {filtered.length === 0 && !loading && (
        <div className="aurelius-card text-center py-12 text-[#9e9eb0]">
          <Search size={32} className="mx-auto mb-3 opacity-40" />
          <p>No workflows match your search.</p>
        </div>
      )}

      {/* Detail Modal */}
      {selected && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
          <div className="bg-[#1a1a2e] border border-[#2d2d44] rounded-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto shadow-2xl">
            <div className="flex items-center justify-between p-5 border-b border-[#2d2d44]">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-[#4fc3f7]/10 text-[#4fc3f7] flex items-center justify-center border border-[#4fc3f7]/20">
                  <GitBranch size={20} />
                </div>
                <div>
                  <h3 className="text-base font-bold text-[#e0e0e0]">{selected.name}</h3>
                  <p className="text-xs text-[#9e9eb0]">{selected.id}</p>
                </div>
              </div>
              <button
                onClick={() => setSelected(null)}
                className="text-[#9e9eb0] hover:text-[#e0e0e0] transition-colors"
              >
                <X size={20} />
              </button>
            </div>
            <div className="p-5 space-y-4">
              <div className="flex items-center gap-2">
                <StatusBadge status={selected.status} />
                <span className="text-xs text-[#9e9eb0]">
                  {selected.event_count} events · Last run {timeAgo(selected.last_run)}
                </span>
              </div>
              {selected.events.length > 0 ? (
                <div>
                  <h4 className="text-xs font-bold text-[#9e9eb0] uppercase tracking-wider mb-2">
                    Event History
                  </h4>
                  <div className="space-y-2 max-h-64 overflow-y-auto">
                    {selected.events.map((e, i) => (
                      <div
                        key={i}
                        className="flex items-start gap-3 text-sm bg-[#0f0f1a] border border-[#2d2d44] rounded-lg p-3"
                      >
                        <div className="w-2 h-2 rounded-full bg-[#4fc3f7] mt-1.5 shrink-0" />
                        <div className="flex-1 min-w-0">
                          <p className="text-[#e0e0e0] font-medium">{e.type}</p>
                          <p className="text-[#9e9eb0] text-xs">{e.message}</p>
                        </div>
                        <span className="text-[10px] text-[#9e9eb0] shrink-0">
                          {timeAgo(e.timestamp)}
                        </span>
                      </div>
                    ))}
                  </div>
                </div>
              ) : (
                <p className="text-sm text-[#9e9eb0]">No events recorded.</p>
              )}
              <div className="flex gap-2">
                <button
                  onClick={() => triggerWorkflow(selected.id, 'start')}
                  disabled={triggering === selected.id}
                  className="aurelius-btn flex-1 flex items-center justify-center gap-2 disabled:opacity-50"
                >
                  {triggering === selected.id ? (
                    <Loader2 size={16} className="animate-spin" />
                  ) : (
                    <Play size={16} />
                  )}
                  Start
                </button>
                <button
                  onClick={() => triggerWorkflow(selected.id, 'complete')}
                  disabled={triggering === selected.id}
                  className="aurelius-btn flex-1 flex items-center justify-center gap-2 disabled:opacity-50"
                >
                  <CheckCircle2 size={16} />
                  Complete
                </button>
                <button
                  onClick={() => triggerWorkflow(selected.id, 'reset')}
                  disabled={triggering === selected.id}
                  className="aurelius-btn flex-1 flex items-center justify-center gap-2 disabled:opacity-50"
                >
                  <RefreshCw size={16} />
                  Reset
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
