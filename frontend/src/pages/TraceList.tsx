import { useState, type ChangeEvent } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import { History, Search, CheckCircle, XCircle, Clock, Play } from 'lucide-react';
import { useApi } from '../hooks/useApi';
import { Input } from '../components/ui/Input';
import { EmptyState } from '../components/EmptyState';
import { Skeleton } from '../components/Skeleton';

interface TraceSummary {
  id: string; agentName: string; task: string;
  status: string; startedAt: number; stepCount: number;
  totalDuration?: number;
}

const statusStyles: Record<string, { color: string; icon: typeof CheckCircle }> = {
  completed: { color: 'text-emerald-400', icon: CheckCircle },
  failed: { color: 'text-rose-400', icon: XCircle },
  running: { color: 'text-[#4fc3f7]', icon: Play },
  truncated: { color: 'text-amber-400', icon: Clock },
};

export default function TraceList() {
  const navigate = useNavigate();
  const [search, setSearch] = useState('');
  const [statusFilter, setStatusFilter] = useState('');
  const { data, loading } = useApi<{ traces: TraceSummary[]; total: number }>(
    `/traces${statusFilter ? `?status=${statusFilter}` : ''}`,
    { refreshInterval: 5000 }
  );

  const traces = (data?.traces || []).filter(t =>
    !search || t.task.toLowerCase().includes(search.toLowerCase()) || t.agentName.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
          <History size={20} className="text-[#4fc3f7]" />
          Execution Traces
          <span className="text-sm font-normal text-[#9e9eb0]">({data?.total || 0} total)</span>
        </h2>
        <div className="flex gap-2">
          <div className="relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#9e9eb0]" />
            <Input
              value={search}
              onChange={(e: ChangeEvent<HTMLInputElement>) => setSearch(e.target.value)}
              placeholder="Search traces..."
              className="pl-8 py-1.5 text-sm w-48"
            />
          </div>
          <select
            value={statusFilter}
            onChange={(e: ChangeEvent<HTMLSelectElement>) => setStatusFilter(e.target.value)}
            className="bg-[#0f0f1a] border border-[#2d2d44] rounded-lg px-3 py-1.5 text-sm text-[#e0e0e0]">
            <option value="">All status</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
            <option value="running">Running</option>
            <option value="truncated">Truncated</option>
          </select>
        </div>
      </div>

      {loading && traces.length === 0 && (
        <div className="space-y-3">{[1,2,3].map(i => <Skeleton key={i} className="h-20" />)}</div>
      )}

      {!loading && traces.length === 0 && (
        <EmptyState
          icon={<History size={24} />}
          title="No Traces"
          description="Run an agent task to generate execution traces."
        />
      )}

      <div className="space-y-2">
        {traces.map(trace => {
          const st = statusStyles[trace.status] || statusStyles.completed;
          const Icon = st.icon;
          return (
            <motion.div
              key={trace.id}
              initial={{ opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              onClick={() => navigate(`/traces/${trace.id}`)}
              className="aurelius-card p-4 hover:border-[#4fc3f7]/30 cursor-pointer transition-all"
            >
              <div className="flex items-start justify-between">
                <div className="flex items-start gap-3 min-w-0">
                  <div className={`mt-1 ${st.color}`}><Icon size={16} /></div>
                  <div className="min-w-0">
                    <p className="text-sm font-medium text-[#e0e0e0] truncate">{trace.task}</p>
                    <div className="flex items-center gap-3 mt-1 text-xs text-[#9e9eb0]">
                      <span>{trace.agentName}</span>
                      <span>·</span>
                      <span>{trace.stepCount} steps</span>
                      {trace.totalDuration && <><span>·</span><span>{(trace.totalDuration / 1000).toFixed(1)}s</span></>}
                    </div>
                  </div>
                </div>
                <span className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded-full ${st.color} bg-[#0f0f1a] border border-[#2d2d44] shrink-0`}>
                  {trace.status}
                </span>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
