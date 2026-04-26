import { GitBranch, Play, Clock, CheckCircle2, XCircle, RotateCcw } from 'lucide-react';

interface Workflow {
  id: number;
  name: string;
  description: string;
  status: 'running' | 'completed' | 'failed' | 'idle';
  lastRun: string;
  duration: string;
}

const workflows: Workflow[] = [
  { id: 1, name: 'Daily Backup', description: 'Backup all agent memory layers to persistent store.', status: 'completed', lastRun: '5 min ago', duration: '42s' },
  { id: 2, name: 'Data Ingest', description: 'Ingest new documents into the knowledge base.', status: 'failed', lastRun: '1 hr ago', duration: '—' },
  { id: 3, name: 'Health Check', description: 'Run system diagnostics across all nodes.', status: 'running', lastRun: 'Just now', duration: '12s' },
  { id: 4, name: 'Memory Prune', description: 'Remove stale and expired memory entries.', status: 'idle', lastRun: '12 min ago', duration: '8s' },
  { id: 5, name: 'Log Rotation', description: 'Archive and compress old system logs.', status: 'completed', lastRun: '2 hr ago', duration: '15s' },
];

const statusBadge = (status: Workflow['status']) => {
  switch (status) {
    case 'running':
      return (
        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-aurelius-accent/10 text-aurelius-accent border border-aurelius-accent/20">
          <RotateCcw size={12} className="animate-spin" />
          Running
        </span>
      );
    case 'completed':
      return (
        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
          <CheckCircle2 size={12} />
          Completed
        </span>
      );
    case 'failed':
      return (
        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-red-500/10 text-red-400 border border-red-500/20">
          <XCircle size={12} />
          Failed
        </span>
      );
    default:
      return (
        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-aurelius-border/20 text-aurelius-muted border border-aurelius-border/40">
          <Clock size={12} />
          Idle
        </span>
      );
  }
};

export default function Workflows() {
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-aurelius-text flex items-center gap-2">
          <GitBranch size={20} className="text-aurelius-accent" />
          Workflows
        </h2>
        <button className="aurelius-btn flex items-center gap-2 text-sm">
          <Play size={14} />
          New Run
        </button>
      </div>

      <div className="space-y-2">
        {workflows.map((wf) => (
          <div
            key={wf.id}
            className="aurelius-card flex flex-col sm:flex-row sm:items-center justify-between gap-3 hover:border-aurelius-accent/30 transition-colors"
          >
            <div className="flex items-center gap-4">
              <div className="w-9 h-9 rounded-lg bg-aurelius-accent/10 text-aurelius-accent flex items-center justify-center border border-aurelius-accent/20">
                <GitBranch size={18} />
              </div>
              <div>
                <h3 className="text-sm font-semibold text-aurelius-text">{wf.name}</h3>
                <p className="text-xs text-aurelius-muted">{wf.description}</p>
              </div>
            </div>
            <div className="flex items-center gap-4 sm:gap-6">
              <div className="text-right hidden sm:block">
                <p className="text-xs text-aurelius-muted">Last run</p>
                <p className="text-sm text-aurelius-text">{wf.lastRun}</p>
              </div>
              <div className="text-right hidden sm:block">
                <p className="text-xs text-aurelius-muted">Duration</p>
                <p className="text-sm text-aurelius-text">{wf.duration}</p>
              </div>
              {statusBadge(wf.status)}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
