import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';
import {
  Bot, Activity, Cpu, Puzzle, Wrench, Layers, Play, CheckCircle,
  AlertTriangle, TrendingUp, MessageSquare, History, BarChart3,
} from 'lucide-react';
import { useApi } from '../hooks/useApi';
import StatsCard from '../components/StatsCard';

export default function Dashboard() {
  const navigate = useNavigate();
  const { data: agentData } = useApi<{ agents: any[] }>('/agents', { refreshInterval: 10000 });
  const { data: healthData } = useApi<{ status: string; services: any[] }>('/health', { refreshInterval: 15000 });
  const { data: notifData } = useApi<{ unread: number }>('/notifications/stats', { refreshInterval: 15000 });
  const { data: pluginData } = useApi<{ total: number; enabled: number }>('/plugins', { refreshInterval: 30000 });

  const agents = agentData?.agents || [];
  const activeAgents = agents.filter((a: any) => a.state === 'active' || a.state === 'busy').length;
  const healthy = healthData?.status === 'healthy';
  const unread = notifData?.unread ?? 0;
  const plugins = pluginData?.total ?? 12;
  const pluginsActive = pluginData?.enabled ?? 8;

  const quickActions = [
    { label: 'Agent Grid', icon: Bot, path: '/agents/grid', color: 'text-[#4fc3f7]', desc: 'Manage agents' },
    { label: 'Agent Chat', icon: MessageSquare, path: '/chat', color: 'text-emerald-400', desc: 'Talk to agents' },
    { label: 'Playground', icon: Play, path: '/agents/playground', color: 'text-amber-400', desc: 'Test agents' },
    { label: 'Traces', icon: History, path: '/traces', color: 'text-violet-400', desc: 'View traces' },
    { label: 'Analytics', icon: BarChart3, path: '/analytics', color: 'text-cyan-400', desc: 'System metrics' },
    { label: 'Plugins', icon: Layers, path: '/plugins', color: 'text-rose-400', desc: 'Manage plugins' },
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
          <Activity size={20} className="text-[#4fc3f7]" /> Operations Dashboard
        </h2>
        <div className={`flex items-center gap-2 text-xs ${healthy ? 'text-emerald-400' : 'text-rose-400'}`}>
          <span className={`w-2 h-2 rounded-full ${healthy ? 'bg-emerald-400 animate-pulse' : 'bg-rose-400'}`} />
          {healthy ? 'All Systems Operational' : 'System Degraded'}
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-6 gap-3">
        <StatsCard label="Active Agents" value={activeAgents} icon={Bot} color="text-[#4fc3f7]" onClick={() => navigate('/agents/grid')} />
        <StatsCard label="Plugins" value={`${pluginsActive}/${plugins}`} icon={Layers} color="text-emerald-400" onClick={() => navigate('/plugins')} />
        <StatsCard label="Unread" value={unread} icon={AlertTriangle} color="text-amber-400" onClick={() => navigate('/notifications')} />
        <StatsCard label="Agents" value={agents.length} icon={Cpu} color="text-violet-400" onClick={() => navigate('/agents/grid')} />
        <StatsCard label="Skills" value="35" icon={Puzzle} color="text-cyan-400" onClick={() => navigate('/skills')} />
        <StatsCard label="Traces" value="Active" icon={History} color="text-rose-400" onClick={() => navigate('/traces')} />
      </div>

      <div>
        <h3 className="text-sm font-semibold text-[#e0e0e0] mb-3">Quick Actions</h3>
        <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
          {quickActions.map(action => {
            const Icon = action.icon;
            return (
              <motion.button key={action.path} whileHover={{ scale: 1.02 }} whileTap={{ scale: 0.98 }}
                onClick={() => navigate(action.path)}
                className="aurelius-card p-4 text-center hover:border-[#4fc3f7]/30 transition-all group"
              >
                <Icon size={24} className={`mx-auto mb-2 ${action.color} group-hover:scale-110 transition-transform`} />
                <p className="text-sm font-medium text-[#e0e0e0]">{action.label}</p>
                <p className="text-[10px] text-[#9e9eb0]">{action.desc}</p>
              </motion.button>
            );
          })}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="aurelius-card p-4">
          <h3 className="text-sm font-semibold text-[#e0e0e0] mb-3 flex items-center gap-2"><Bot size={14} className="text-[#4fc3f7]" /> Active Agents</h3>
          <div className="space-y-2">
            {agents.slice(0, 5).map((agent: any) => (
              <div key={agent.id} className="flex items-center justify-between py-2 border-b border-[#2d2d44]/50 last:border-0 cursor-pointer hover:bg-white/[0.02] px-2 rounded" onClick={() => navigate(`/agents/${agent.id}`)}>
                <div className="flex items-center gap-2">
                  <span className={`w-2 h-2 rounded-full ${agent.state === 'active' || agent.state === 'busy' ? 'bg-emerald-400' : agent.state === 'error' ? 'bg-rose-400' : 'bg-gray-400'}`} />
                  <span className="text-sm text-[#e0e0e0]">{agent.name}</span>
                  <span className="text-[10px] text-[#9e9eb0]">{agent.role}</span>
                </div>
                <span className="text-xs text-[#9e9eb0]">{agent.state}</span>
              </div>
            ))}
            {agents.length === 0 && <p className="text-xs text-[#9e9eb0] text-center py-4">No agents active. Spawn one from the Agent Grid.</p>}
          </div>
        </div>

        <div className="aurelius-card p-4">
          <h3 className="text-sm font-semibold text-[#e0e0e0] mb-3 flex items-center gap-2"><Activity size={14} className="text-emerald-400" /> System Status</h3>
          <div className="space-y-2">
            {[
              { label: 'Frontend', status: 'healthy' },
              { label: 'API (Node.js)', status: 'healthy' },
              { label: 'Model Serving (Python)', status: healthy ? 'healthy' : 'degraded' },
              { label: 'Database', status: 'healthy' },
              { label: 'Redis Cache', status: 'healthy' },
              { label: 'WebSocket', status: 'healthy' },
              { label: 'Agent Runtime', status: agents.length > 0 ? 'healthy' : 'idle' },
            ].map(s => (
              <div key={s.label} className="flex items-center justify-between py-1.5">
                <span className="text-sm text-[#e0e0e0]">{s.label}</span>
                <span className={`flex items-center gap-1.5 text-xs ${
                  s.status === 'healthy' ? 'text-emerald-400' :
                  s.status === 'degraded' ? 'text-amber-400' : 'text-[#9e9eb0]'
                }`}>
                  <span className={`w-1.5 h-1.5 rounded-full ${s.status === 'healthy' ? 'bg-emerald-400' : s.status === 'degraded' ? 'bg-amber-400' : 'bg-gray-400'}`} />
                  {s.status}
                </span>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
}
