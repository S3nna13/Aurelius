import { useState } from 'react';
import {
  CalendarClock,
  Play,
  Pause,
  RotateCcw,
  CheckCircle2,
  AlertTriangle,
  Clock,
  Loader2,
} from 'lucide-react';
import { useToast } from '../components/ToastProvider';

interface ScheduledTask {
  id: string;
  name: string;
  schedule: string;
  status: 'active' | 'paused' | 'running' | 'failed';
  lastRun: string;
  nextRun: string;
  description: string;
}

const mockTasks: ScheduledTask[] = [
  {
    id: 'backup-daily',
    name: 'Daily Backup',
    schedule: '0 2 * * *',
    status: 'active',
    lastRun: '2 hr ago',
    nextRun: 'in 22 hr',
    description: 'Full system backup to remote storage.',
  },
  {
    id: 'health-check',
    name: 'Health Check',
    schedule: '*/15 * * * *',
    status: 'active',
    lastRun: '8 min ago',
    nextRun: 'in 7 min',
    description: 'Monitor agent health and resource usage.',
  },
  {
    id: 'memory-cleanup',
    name: 'Memory Cleanup',
    schedule: '0 4 * * 0',
    status: 'paused',
    lastRun: '3 days ago',
    nextRun: 'Paused',
    description: 'Archive old session data and compact layers.',
  },
  {
    id: 'model-sync',
    name: 'Model Sync',
    schedule: '0 */6 * * *',
    status: 'running',
    lastRun: 'Just now',
    nextRun: 'Running...',
    description: 'Synchronize model weights with upstream.',
  },
];

const statusConfig: Record<string, { icon: typeof Play; color: string; bg: string; border: string; label: string }> = {
  active: { icon: Play, color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/20', label: 'Active' },
  paused: { icon: Pause, color: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/20', label: 'Paused' },
  running: { icon: Loader2, color: 'text-[#4fc3f7]', bg: 'bg-[#4fc3f7]/10', border: 'border-[#4fc3f7]/20', label: 'Running' },
  failed: { icon: AlertTriangle, color: 'text-rose-400', bg: 'bg-rose-500/10', border: 'border-rose-500/20', label: 'Failed' },
};

export default function ScheduledTasks() {
  const [tasks, setTasks] = useState<ScheduledTask[]>(mockTasks);
  const { toast } = useToast();

  const toggleTask = (id: string) => {
    setTasks((prev) =>
      prev.map((t) =>
        t.id === id
          ? { ...t, status: t.status === 'paused' ? 'active' : 'paused' as ScheduledTask['status'] }
          : t
      )
    );
    toast('Task status updated', 'success');
  };

  const runNow = (id: string) => {
    toast(`Triggering ${id}...`, 'info');
  };

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
          <CalendarClock size={20} className="text-[#4fc3f7]" />
          Scheduled Tasks
        </h2>
        <div className="flex gap-2 text-sm text-[#9e9eb0]">
          <span className="flex items-center gap-1">
            <CheckCircle2 size={14} className="text-emerald-400" />
            {tasks.filter((t) => t.status === 'active').length} active
          </span>
          <span className="flex items-center gap-1">
            <Pause size={14} className="text-amber-400" />
            {tasks.filter((t) => t.status === 'paused').length} paused
          </span>
        </div>
      </div>

      <div className="space-y-3">
        {tasks.map((task) => {
          const s = statusConfig[task.status];
          const Icon = s.icon;
          return (
            <div key={task.id} className="aurelius-card space-y-3">
              <div className="flex items-start justify-between">
                <div className="flex items-center gap-3">
                  <div className={`w-9 h-9 rounded-lg flex items-center justify-center border ${s.bg} ${s.color} ${s.border}`}>
                    <Icon size={16} className={task.status === 'running' ? 'animate-spin' : ''} />
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-[#e0e0e0]">{task.name}</h3>
                    <p className="text-xs text-[#9e9eb0]">{task.description}</p>
                  </div>
                </div>
                <span className={`text-xs font-bold px-2 py-0.5 rounded-full border ${s.bg} ${s.color} ${s.border}`}>
                  {s.label}
                </span>
              </div>

              <div className="flex flex-wrap items-center gap-4 text-xs text-[#9e9eb0]">
                <span className="flex items-center gap-1 font-mono bg-[#0f0f1a] px-2 py-0.5 rounded border border-[#2d2d44]">
                  <Clock size={10} />
                  {task.schedule}
                </span>
                <span>Last: {task.lastRun}</span>
                <span>Next: {task.nextRun}</span>
              </div>

              <div className="flex gap-2 pt-1">
                <button
                  onClick={() => toggleTask(task.id)}
                  className="aurelius-btn-outline text-xs flex items-center gap-1.5"
                >
                  {task.status === 'paused' ? <Play size={12} /> : <Pause size={12} />}
                  {task.status === 'paused' ? 'Resume' : 'Pause'}
                </button>
                <button
                  onClick={() => runNow(task.id)}
                  className="aurelius-btn-outline text-xs flex items-center gap-1.5"
                >
                  <RotateCcw size={12} />
                  Run Now
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
