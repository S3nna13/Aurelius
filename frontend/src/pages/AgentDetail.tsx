import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  ArrowLeft, Bot, Activity, Clock, CheckCircle, XCircle,
  Cpu, Terminal, MessageSquare, Wrench, Puzzle, Globe,
} from 'lucide-react';

interface AgentDetailData {
  id: string; name: string; role: string; state: string;
  capabilities: string[]; created: number;
  metrics: { tasksCompleted: number; avgLatencyMs: number; errorRate: number; tokensUsed: number; uptimeHours: number };
}

export default function AgentDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [agent, setAgent] = useState<AgentDetailData | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetch(`/api/agents/${id}`)
      .then(r => r.json())
      .then(d => { setAgent(d.agent); setLoading(false); })
      .catch(() => setLoading(false));
  }, [id]);

  if (loading) return <div className="flex justify-center py-20"><div className="w-8 h-8 border-2 border-[#4fc3f7] border-t-transparent rounded-full animate-spin" /></div>;
  if (!agent) return <div className="text-center py-20 text-[#9e9eb0]">Agent not found</div>;

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="flex items-center gap-3">
        <button onClick={() => navigate('/agents/grid')} className="text-[#9e9eb0] hover:text-[#e0e0e0]"><ArrowLeft size={20} /></button>
        <div className="w-10 h-10 rounded-lg bg-[#4fc3f7]/10 flex items-center justify-center"><Bot size={20} className="text-[#4fc3f7]" /></div>
        <div>
          <h2 className="text-lg font-bold text-[#e0e0e0]">{agent.name}</h2>
          <p className="text-xs text-[#9e9eb0]">{agent.role} · ID: {agent.id}</p>
        </div>
        <span className={`ml-auto text-xs font-bold px-3 py-1 rounded-full border ${
          agent.state === 'active' ? 'text-emerald-400 border-emerald-500/20 bg-emerald-500/10' :
          agent.state === 'busy' ? 'text-amber-400 border-amber-500/20 bg-amber-500/10' :
          agent.state === 'error' ? 'text-rose-400 border-rose-500/20 bg-rose-500/10' :
          'text-[#9e9eb0] border-[#2d2d44] bg-[#0f0f1a]'
        }`}>{agent.state}</span>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: 'Tasks', value: agent.metrics.tasksCompleted, icon: CheckCircle, color: 'text-emerald-400' },
          { label: 'Avg Latency', value: `${agent.metrics.avgLatencyMs.toFixed(0)}ms`, icon: Clock, color: 'text-amber-400' },
          { label: 'Error Rate', value: `${(agent.metrics.errorRate * 100).toFixed(1)}%`, icon: XCircle, color: 'text-rose-400' },
          { label: 'Tokens Used', value: (agent.metrics.tokensUsed || 0).toLocaleString(), icon: Cpu, color: 'text-[#4fc3f7]' },
        ].map(s => (
          <div key={s.label} className="aurelius-card text-center py-3">
            <s.icon size={16} className={`mx-auto mb-1 ${s.color}`} />
            <p className={`text-lg font-bold ${s.color}`}>{s.value}</p>
            <p className="text-[10px] text-[#9e9eb0] uppercase tracking-wider">{s.label}</p>
          </div>
        ))}
      </div>

      <div className="aurelius-card p-4">
        <h3 className="text-sm font-semibold text-[#e0e0e0] mb-3">Capabilities</h3>
        <div className="flex flex-wrap gap-2">
          {agent.capabilities.map(cap => (
            <span key={cap} className="text-xs text-[#4fc3f7] bg-[#4fc3f7]/10 border border-[#4fc3f7]/20 px-3 py-1 rounded-lg">{cap}</span>
          ))}
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        <div className="aurelius-card p-4">
          <h3 className="text-sm font-semibold text-[#e0e0e0] mb-3 flex items-center gap-2"><Activity size={14} className="text-[#4fc3f7]" /> Recent Activity</h3>
          <div className="space-y-2 text-xs text-[#9e9eb0]">
            {[1,2,3,4,5].map(i => (
              <div key={i} className="flex items-center gap-2 py-1 border-b border-[#2d2d44]/50 last:border-0">
                <Terminal size={10} className="text-[#4fc3f7]/60" />
                <span>Task executed {i}0 min ago</span>
              </div>
            ))}
          </div>
        </div>
        <div className="aurelius-card p-4">
          <h3 className="text-sm font-semibold text-[#e0e0e0] mb-3 flex items-center gap-2"><Wrench size={14} className="text-amber-400" /> Available Tools</h3>
          <div className="flex flex-wrap gap-2">
            {['read_file', 'write_file', 'search_web', 'run_command', 'query_db'].map(tool => (
              <span key={tool} className="text-xs text-amber-400 bg-amber-500/10 border border- amber-500/20 px-2 py-1 rounded">{tool}</span>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
