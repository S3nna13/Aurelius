import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  Cpu, Play, Square, AlertCircle, CheckCircle, Clock, Activity,
  Plus, RefreshCw, Loader2, Terminal, MessageSquare, GitBranch,
} from 'lucide-react';
import { useApi } from '../hooks/useApi';
import StatsCard from '../components/StatsCard';
import Skeleton from '../components/Skeleton';
import EmptyState from '../components/EmptyState';

interface Agent {
  id: string;
  name: string;
  role: string;
  capabilities: string[];
  state: 'idle' | 'active' | 'busy' | 'error' | 'terminated';
  created: number;
  lastHeartbeat: number;
  metrics: { tasksCompleted: number; avgLatencyMs: number; errorRate: number };
}

const stateStyles: Record<string, { color: string; bg: string; icon: typeof Cpu }> = {
  idle: { color: 'text-gray-400', bg: 'bg-gray-500/10', icon: Clock },
  active: { color: 'text-emerald-400', bg: 'bg-emerald-500/10', icon: Activity },
  busy: { color: 'text-amber-400', bg: 'bg-amber-500/10', icon: Cpu },
  error: { color: 'text-rose-400', bg: 'bg-rose-500/10', icon: AlertCircle },
  terminated: { color: 'text-gray-600', bg: 'bg-gray-500/5', icon: Square },
};

function AgentCard({ agent, onTerminate, onSelect }: {
  agent: Agent;
  onTerminate: (id: string) => void;
  onSelect: (id: string) => void;
}) {
  const style = stateStyles[agent.state] || stateStyles.idle;
  const Icon = style.icon;
  const age = Math.floor((Date.now() - agent.created) / 1000);
  const ageStr = age < 60 ? `${age}s` : age < 3600 ? `${Math.floor(age / 60)}m` : `${Math.floor(age / 3600)}h`;

  return (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      className="aurelius-card p-4 hover:border-[#4fc3f7]/30 transition-all cursor-pointer group"
      onClick={() => onSelect(agent.id)}
    >
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center gap-3">
          <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${style.bg} ${style.color}`}>
            <Icon size={20} />
          </div>
          <div>
            <h3 className="font-semibold text-[#e0e0e0] text-sm">{agent.name}</h3>
            <p className="text-xs text-[#9e9eb0]">{agent.role}</p>
          </div>
        </div>
        <button
          onClick={(e) => { e.stopPropagation(); onTerminate(agent.id); }}
          className="opacity-0 group-hover:opacity-100 text-rose-400 hover:text-rose-300 transition-all p-1"
          title="Terminate agent"
        >
          <Square size={14} />
        </button>
      </div>

      <div className="flex items-center gap-2 mb-3 flex-wrap">
        <span className={`text-[10px] uppercase font-bold px-2 py-0.5 rounded-full ${style.bg} ${style.color} border ${style.bg.replace('bg-', 'border-').replace('/10', '/20')}`}>
          {agent.state}
        </span>
        <span className="text-[10px] text-[#9e9eb0] bg-[#0f0f1a] px-2 py-0.5 rounded-full border border-[#2d2d44]">
          {ageStr} ago
        </span>
      </div>

      <div className="grid grid-cols-3 gap-2 text-center">
        <div>
          <p className="text-sm font-bold text-[#e0e0e0]">{agent.metrics.tasksCompleted}</p>
          <p className="text-[9px] text-[#9e9eb0] uppercase tracking-wider">Tasks</p>
        </div>
        <div>
          <p className="text-sm font-bold text-[#e0e0e0]">{agent.metrics.avgLatencyMs.toFixed(0)}ms</p>
          <p className="text-[9px] text-[#9e9eb0] uppercase tracking-wider">Latency</p>
        </div>
        <div>
          <p className={`text-sm font-bold ${agent.metrics.errorRate > 0.1 ? 'text-rose-400' : 'text-[#e0e0e0]'}`}>
            {(agent.metrics.errorRate * 100).toFixed(0)}%
          </p>
          <p className="text-[9px] text-[#9e9eb0] uppercase tracking-wider">Errors</p>
        </div>
      </div>
    </motion.div>
  );
}

export default function AgentGrid() {
  const navigate = useNavigate();
  const { data, loading, refresh } = useApi<{ agents: Agent[] }>('/agents', { refreshInterval: 3000 });
  const [deleting, setDeleting] = useState<string | null>(null);

  const agents = data?.agents || [];

  const handleTerminate = useCallback(async (id: string) => {
    setDeleting(id);
    try {
      await fetch(`/api/agents/${id}`, { method: 'DELETE' });
    } catch { /* ignore */ }
    setDeleting(null);
    refresh();
  }, [refresh]);

  const handleSpawn = () => navigate('/agents/new');

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
            <Cpu size={20} className="text-[#4fc3f7]" />
            Agent Operations Center
          </h2>
          <p className="text-sm text-[#9e9eb0] mt-0.5">
            {agents.length} agent{agents.length !== 1 ? 's' : ''} · {agents.filter(a => a.state === 'active' || a.state === 'busy').length} active
          </p>
        </div>
        <div className="flex gap-2">
          <button onClick={refresh} className="aurelius-btn-outline flex items-center gap-2 text-sm">
            <RefreshCw size={14} />
            Refresh
          </button>
          <button onClick={handleSpawn} className="aurelius-btn-primary flex items-center gap-2 text-sm">
            <Plus size={14} />
            Spawn Agent
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatsCard label="Total Agents" value={agents.length} icon={Cpu} color="text-[#4fc3f7]" />
        <StatsCard label="Active" value={agents.filter(a => a.state === 'active' || a.state === 'busy').length} icon={Activity} color="text-emerald-400" />
        <StatsCard label="Idle" value={agents.filter(a => a.state === 'idle').length} icon={Clock} color="text-amber-400" />
        <StatsCard label="Errors" value={agents.filter(a => a.state === 'error').length} icon={AlertCircle} color="text-rose-400" />
      </div>

      {loading && agents.length === 0 && (
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {[1,2,3].map(i => <Skeleton key={i} className="h-40" />)}
        </div>
      )}

      {!loading && agents.length === 0 && (
        <EmptyState
          icon={Cpu}
          title="No Agents Running"
          description="Spawn your first agent to get started with automated task execution."
          action={{ label: 'Spawn Agent', onClick: handleSpawn }}
        />
      )}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {agents.map(agent => (
          <AgentCard
            key={agent.id}
            agent={agent}
            onTerminate={handleTerminate}
            onSelect={(id) => navigate(`/agents/${id}`)}
          />
        ))}
      </div>
    </div>
  );
}
