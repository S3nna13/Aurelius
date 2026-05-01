import { useState } from 'react';
import { motion } from 'framer-motion';
import { CalendarClock, Play, Square, Plus, RefreshCw, Clock, CheckCircle, XCircle, Trash2 } from 'lucide-react';
import { useApi } from '../hooks/useApi';
import EmptyState from '../components/EmptyState';
import Toggle from '../components/ui/Toggle';

interface ScheduledTask { id: string; name: string; cron: string; task: string; enabled: boolean; lastRun?: string; lastStatus?: string; nextRun?: string; }

export default function ScheduledTasks() {
  const { data, loading, refresh } = useApi<{ tasks: ScheduledTask[] }>('/scheduler', { refreshInterval: 10000 });
  const tasks = data?.tasks || [];

  const toggleTask = async (id: string, enabled: boolean) => {
    await fetch(`/api/scheduler/${id}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ enabled: !enabled }) });
    refresh();
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2"><CalendarClock size={20} className="text-[#4fc3f7]" />Scheduled Tasks</h2>
        <button className="aurelius-btn-primary flex items-center gap-2 text-sm"><Plus size={14} /> New Task</button>
      </div>
      {tasks.length === 0 && !loading && <EmptyState icon={CalendarClock} title="No Scheduled Tasks" description="Schedule agent tasks to run automatically." />}
      <div className="space-y-2">
        {tasks.map(task => (
          <motion.div key={task.id} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="aurelius-card p-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-[#4fc3f7]/10 flex items-center justify-center"><CalendarClock size={16} className="text-[#4fc3f7]" /></div>
              <div><p className="text-sm font-medium text-[#e0e0e0]">{task.name}</p>
                <div className="flex items-center gap-2 text-xs text-[#9e9eb0] mt-0.5">
                  <span className="font-mono">{task.cron}</span><span>·</span><span>{task.task}</span>
                  {task.nextRun && <><span>·</span><span>Next: {new Date(task.nextRun).toLocaleTimeString()}</span></>}
                </div>
              </div>
            </div>
            <div className="flex items-center gap-3">
              {task.lastStatus === 'completed' && <CheckCircle size={14} className="text-emerald-400" />}
              {task.lastStatus === 'failed' && <XCircle size={14} className="text-rose-400" />}
              <Toggle checked={task.enabled} onChange={() => toggleTask(task.id, task.enabled)} />
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
