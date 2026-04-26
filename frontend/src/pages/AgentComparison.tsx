import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Bot,
  Activity,
  HeartPulse,
  Zap,
  Pause,
  Play,
  ArrowLeft,
  BarChart3,
} from 'lucide-react';
import { useApi } from '../hooks/useApi';

interface AgentStatus {
  id: string;
  state: string;
}

const agentRoles: Record<string, string> = {
  hermes: 'Notification Router',
  openclaw: 'Task Orchestrator',
  cerebrum: 'Memory Manager',
  vigil: 'Security Warden',
  default: 'System Agent',
};

const mockMetrics: Record<string, Record<string, number>> = {
  hermes: { messages_routed: 1240, uptime_pct: 99.8, latency_ms: 12 },
  openclaw: { tasks_completed: 342, uptime_pct: 99.5, latency_ms: 45 },
  cerebrum: { queries_served: 8901, uptime_pct: 99.9, latency_ms: 28 },
  vigil: { alerts_processed: 567, uptime_pct: 99.7, latency_ms: 8 },
};

export default function AgentComparison() {
  const navigate = useNavigate();
  const [selected, setSelected] = useState<string[]>([]);

  const { data } = useApi<{ agents: AgentStatus[] }>('/status', {
    refreshInterval: 10000,
  });

  const agents = data?.agents || [];

  const toggleAgent = (id: string) => {
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((a) => a !== id) : prev.length < 3 ? [...prev, id] : prev
    );
  };

  const compared = agents.filter((a) => selected.includes(a.id));

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <button
          onClick={() => navigate('/')}
          className="p-2 rounded-lg text-aurelius-muted hover:text-aurelius-text hover:bg-aurelius-border/40 transition-colors"
        >
          <ArrowLeft size={18} />
        </button>
        <h2 className="text-lg font-bold text-aurelius-text flex items-center gap-2">
          <BarChart3 size={20} className="text-aurelius-accent" />
          Agent Comparison
        </h2>
      </div>

      <p className="text-sm text-aurelius-muted">
        Select up to 3 agents to compare side-by-side.
      </p>

      {/* Agent Selector */}
      <div className="flex flex-wrap gap-2">
        {agents.map((agent) => {
          const isSelected = selected.includes(agent.id);
          const stateUpper = agent.state.toUpperCase();
          const isActive = stateUpper === 'ACTIVE' || stateUpper === 'RUNNING';
          return (
            <button
              key={agent.id}
              onClick={() => toggleAgent(agent.id)}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg text-xs font-medium border transition-colors ${
                isSelected
                  ? 'bg-aurelius-accent/10 text-aurelius-accent border-aurelius-accent/30'
                  : 'bg-aurelius-bg text-aurelius-muted border-aurelius-border hover:border-aurelius-accent/20'
              }`}
            >
              {isActive ? <Play size={12} className="text-emerald-400" /> : <Pause size={12} className="text-amber-400" />}
              {agent.id}
            </button>
          );
        })}
      </div>

      {/* Comparison Grid */}
      {compared.length > 0 && (
        <div className={`grid gap-4 ${compared.length === 1 ? 'grid-cols-1' : compared.length === 2 ? 'grid-cols-1 sm:grid-cols-2' : 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3'}`}>
          {compared.map((agent) => {
            const metrics = mockMetrics[agent.id.toLowerCase()] || {};
            const stateUpper = agent.state.toUpperCase();
            const isActive = stateUpper === 'ACTIVE' || stateUpper === 'RUNNING';
            return (
              <div key={agent.id} className="aurelius-card space-y-4">
                <div className="flex items-center gap-3">
                  <div className="w-10 h-10 rounded-full bg-aurelius-accent/10 text-aurelius-accent flex items-center justify-center border border-aurelius-accent/20">
                    <Bot size={18} />
                  </div>
                  <div>
                    <h3 className="text-sm font-bold text-aurelius-text">
                      {agent.id.replace(/^\w/, (c) => c.toUpperCase())}
                    </h3>
                    <p className="text-xs text-aurelius-muted">
                      {agentRoles[agent.id.toLowerCase()] || agentRoles.default}
                    </p>
                  </div>
                </div>

                <div className="space-y-2">
                  <div className="flex items-center justify-between py-2 border-b border-aurelius-border/50">
                    <span className="text-xs text-aurelius-muted flex items-center gap-1">
                      <Activity size={12} /> State
                    </span>
                    <span className={`text-xs font-bold ${isActive ? 'text-emerald-400' : 'text-amber-400'}`}>
                      {agent.state}
                    </span>
                  </div>
                  <div className="flex items-center justify-between py-2 border-b border-aurelius-border/50">
                    <span className="text-xs text-aurelius-muted flex items-center gap-1">
                      <HeartPulse size={12} /> Uptime
                    </span>
                    <span className="text-xs font-bold text-aurelius-text">
                      {metrics.uptime_pct?.toFixed(1) ?? '—'}%
                    </span>
                  </div>
                  <div className="flex items-center justify-between py-2 border-b border-aurelius-border/50">
                    <span className="text-xs text-aurelius-muted flex items-center gap-1">
                      <Zap size={12} /> Latency
                    </span>
                    <span className="text-xs font-bold text-aurelius-text">
                      {metrics.latency_ms ?? '—'} ms
                    </span>
                  </div>
                  {Object.entries(metrics).filter(([k]) => !['uptime_pct', 'latency_ms'].includes(k)).map(([key, value]) => (
                    <div key={key} className="flex items-center justify-between py-2 border-b border-aurelius-border/50">
                      <span className="text-xs text-aurelius-muted capitalize">{key.replace(/_/g, ' ')}</span>
                      <span className="text-xs font-bold text-aurelius-text">{value.toLocaleString()}</span>
                    </div>
                  ))}
                </div>

                <button
                  onClick={() => navigate(`/agents/${agent.id}`)}
                  className="w-full aurelius-btn-outline text-xs"
                >
                  View Details
                </button>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
