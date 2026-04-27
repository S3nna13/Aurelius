import { useState, useEffect, useCallback } from 'react';
import { Link } from 'react-router-dom';
import {
  Bot, CheckCircle2, Wrench, HeartPulse, Activity, Zap, Pause, Play,
  Terminal, Loader2, RefreshCw, Clock, Download, ChevronRight,
} from 'lucide-react';
import { LineChart, BarChart, DonutChart } from '../components/charts';
import { downloadJSON } from '../utils/export';
import { api } from '../api/AureliusClient';
import { useAgentStore } from '../stores/agentStore';
import { useApiStore } from '../stores/apiStore';

interface ActivityEntry {
  id: string;
  timestamp: number;
  command: string;
  success: boolean;
  output: string;
}

const agentRoles: Record<string, string> = {
  hermes: 'Notification Router',
  openclaw: 'Task Orchestrator',
  cerebrum: 'Memory Manager',
  vigil: 'Security Warden',
  default: 'System Agent',
};

const agentIcon = (state: string) => {
  const s = state.toUpperCase();
  if (s === 'ACTIVE' || s === 'RUNNING') return <Play size={14} className="text-emerald-400" />;
  if (s === 'IDLE') return <Pause size={14} className="text-amber-400" />;
  return <Zap size={14} className="text-rose-400" />;
};

function timeAgo(ts: number): string {
  const diff = Date.now() / 1000 - ts;
  if (diff < 60) return 'Just now';
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} hr ago`;
  return `${Math.floor(diff / 86400)} days ago`;
}

export default function Dashboard() {

  const agents = useAgentStore((s) => s.agents);
  const health = useApiStore((s) => s.health);
  const [activity, setActivity] = useState<ActivityEntry[]>([]);
  const [activityLoading, setActivityLoading] = useState(true);
  const [activityError, setActivityError] = useState<Error | null>(null);

  const fetchActivity = useCallback(async () => {
    try {
      const res = await api.getActivity(100);
      setActivity(res.entries);
      setActivityError(null);
    } catch (err) {
      setActivityError(err instanceof Error ? err : new Error(String(err)));
    } finally {
      setActivityLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchActivity();
    const interval = setInterval(fetchActivity, 15000);
    return () => clearInterval(interval);
  }, [fetchActivity]);

  const counts = {
    agents_online: agents.filter(a =>
      ['ACTIVE', 'RUNNING', 'IDLE'].includes(a.state.toUpperCase())
    ).length,
    agents_total: agents.length,
    skills_active: 2,
    skills_total: 3,
    plugins_total: 1,
    notifications_unread: 0,
  };

  const healthPercent = counts.agents_total > 0
    ? Math.round((counts.agents_online / counts.agents_total) * 100)
    : 100;

  const statusCards = [
    {
      label: 'Agents Online', value: `${counts.agents_online}/${counts.agents_total}`,
      icon: Bot, color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/20',
    },
    {
      label: 'Skills Active', value: `${counts.skills_active}/${counts.skills_total}`,
      icon: Wrench, color: 'text-[#4fc3f7]', bg: 'bg-[#4fc3f7]/10', border: 'border-[#4fc3f7]/20',
    },
    {
      label: 'Plugins Loaded', value: `${counts.plugins_total}`,
      icon: Terminal, color: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/20',
    },
    {
      label: 'System Health', value: `${healthPercent}%`,
      icon: HeartPulse,
      color: healthPercent >= 80 ? 'text-emerald-400' : healthPercent >= 50 ? 'text-amber-400' : 'text-rose-400',
      bg: healthPercent >= 80 ? 'bg-emerald-500/10' : healthPercent >= 50 ? 'bg-amber-500/10' : 'bg-rose-500/10',
      border: healthPercent >= 80 ? 'border-emerald-500/20' : healthPercent >= 50 ? 'border-amber-500/20' : 'border-rose-500/20',
    },
  ];

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
          <Activity size={20} className="text-[#4fc3f7]" />
          Mission Control
          {agents.length > 0 && <span className="text-[10px] font-bold text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20 ml-2">LIVE</span>}
        </h2>
        <div className="flex gap-2 self-start sm:self-auto">
          <button
            onClick={() => downloadJSON(
              { agents, activity, exportedAt: new Date().toISOString() },
              `aurelius-dashboard-${Date.now()}.json`,
            )}
            className="aurelius-btn-outline flex items-center gap-2 text-sm"
          >
            <Download size={14} /> Export
          </button>
          <button
            onClick={fetchActivity}
            disabled={activityLoading}
            className="aurelius-btn-outline flex items-center gap-2 text-sm disabled:opacity-50"
          >
            <RefreshCw size={14} /> Refresh
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4">
        {statusCards.map((card) => (
          <div key={card.label} className="aurelius-card space-y-2">
            <div className="flex items-center justify-between">
              <span className={card.color}><card.icon size={20} /></span>
              <span className={`text-xs font-bold px-2 py-0.5 rounded border ${card.bg} ${card.color} ${card.border}`}>Live</span>
            </div>
            <p className={`text-2xl font-bold ${card.color}`}>{card.value}</p>
            <p className="text-xs text-[#9e9eb0] uppercase tracking-wider">{card.label}</p>
          </div>
        ))}
      </div>

      {health && (
        <div className="aurelius-card flex items-center gap-3 py-3 px-4 border-emerald-500/10">
          <HeartPulse size={16} className="text-emerald-400" />
          <span className="text-xs text-[#9e9eb0]">Server: <span className="text-emerald-400">{health.status}</span></span>
          <span className="text-xs text-[#9e9eb0]">Uptime: <span className="text-[#e0e0e0]">{(health.uptime / 3600).toFixed(1)}h</span></span>
          {agents.length === 0 && <span className="text-xs text-amber-400 ml-auto">No data</span>}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="aurelius-card space-y-4">
          <h3 className="text-sm font-semibold text-[#e0e0e0] uppercase tracking-wider flex items-center gap-2">
            <Bot size={16} className="text-[#4fc3f7]" /> Active Agents
          </h3>
          <div className="space-y-2">
            {agents.map((agent) => (
              <Link
                key={agent.id}
                to={`/agents/${agent.id}`}
                className="flex items-center justify-between px-3 py-2.5 rounded-lg bg-[#0f0f1a]/50 border border-[#2d2d44]/50 hover:border-[#4fc3f7]/20 transition-colors group"
              >
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-[#4fc3f7]/10 text-[#4fc3f7] flex items-center justify-center border border-[#4fc3f7]/20">
                    <Bot size={14} />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-[#e0e0e0]">{agent.id.replace(/^\w/, (c) => c.toUpperCase())}</p>
                    <p className="text-xs text-[#9e9eb0]">{agentRoles[agent.id.toLowerCase()] || agentRoles.default}</p>
                  </div>
                </div>
                <div className="flex items-center gap-2">
                  {agentIcon(agent.state)}
                  <span className={`text-xs font-bold px-2 py-0.5 rounded-full border ${
                    agent.state.toUpperCase() === 'ACTIVE' || agent.state.toUpperCase() === 'RUNNING'
                      ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                      : agent.state.toUpperCase() === 'IDLE'
                      ? 'bg-amber-500/10 text-amber-400 border-amber-500/20'
                      : 'bg-rose-500/10 text-rose-400 border-rose-500/20'
                  }`}>
                    {agent.state}
                  </span>
                  <ChevronRight size={14} className="text-[#9e9eb0] opacity-0 group-hover:opacity-100 transition-opacity" />
                </div>
              </Link>
            ))}
            {agents.length === 0 && (
              <p className="text-sm text-[#9e9eb0] text-center py-6">No agents reported.</p>
            )}
          </div>
        </div>

        <div className="aurelius-card space-y-4">
          <h3 className="text-sm font-semibold text-[#e0e0e0] uppercase tracking-wider flex items-center gap-2">
            <CheckCircle2 size={16} className="text-[#4fc3f7]" /> Activity Feed
          </h3>
          {activityLoading && activity.length === 0 ? (
            <div className="text-center py-8 text-[#9e9eb0]">
              <Loader2 size={24} className="mx-auto mb-2 animate-spin opacity-60" />
              <p className="text-sm">Loading activity...</p>
            </div>
          ) : (
            <div className="space-y-2 max-h-[400px] overflow-y-auto">
              {activity.slice().reverse().map((entry) => (
                <div key={entry.id} className="flex items-start gap-3 px-3 py-2.5 rounded-lg bg-[#0f0f1a]/50 border border-[#2d2d44]/50 hover:border-[#4fc3f7]/10 transition-colors">
                  <div className={`mt-0.5 w-2 h-2 rounded-full shrink-0 ${entry.success ? 'bg-emerald-400' : 'bg-rose-400'}`} />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm text-[#e0e0e0] truncate">{entry.command}</p>
                    <p className="text-xs text-[#9e9eb0] truncate mt-0.5">{entry.output}</p>
                    <p className="text-[10px] text-[#9e9eb0]/60 mt-1 flex items-center gap-1">
                      <Clock size={10} />{timeAgo(entry.timestamp)}
                    </p>
                  </div>
                </div>
              ))}
              {activity.length === 0 && (
                <p className="text-sm text-[#9e9eb0] text-center py-6">No recent activity.</p>
              )}
            </div>
          )}
          {activityError && <p className="text-xs text-rose-400 text-center">{activityError.message}</p>}
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="aurelius-card space-y-4">
          <h3 className="text-sm font-semibold text-[#e0e0e0] uppercase tracking-wider flex items-center gap-2">
            <Activity size={16} className="text-[#4fc3f7]" /> Activity Trend
          </h3>
          {activity.length > 0 ? (
            <LineChart data={activity.slice(-20).map((e, i) => ({ label: `${i + 1}`, value: e.success ? 1 : 0 }))} color="#4fc3f7" />
          ) : (
            <p className="text-sm text-[#9e9eb0] text-center py-6">No activity data yet.</p>
          )}
        </div>
        <div className="aurelius-card space-y-4">
          <h3 className="text-sm font-semibold text-[#e0e0e0] uppercase tracking-wider flex items-center gap-2">
            <Bot size={16} className="text-[#4fc3f7]" /> Agent Status
          </h3>
          {agents.length > 0 ? (
            <DonutChart data={[
              { label: 'Active', value: agents.filter((a) => a.state.toUpperCase() === 'ACTIVE' || a.state.toUpperCase() === 'RUNNING').length, color: '#34d399' },
              { label: 'Idle', value: agents.filter((a) => a.state.toUpperCase() === 'IDLE').length, color: '#fbbf24' },
              { label: 'Other', value: agents.filter((a) => !['ACTIVE', 'RUNNING', 'IDLE'].includes(a.state.toUpperCase())).length, color: '#f87171' },
            ]} />
          ) : (
            <p className="text-sm text-[#9e9eb0] text-center py-6">No agent data yet.</p>
          )}
        </div>
      </div>

      <div className="aurelius-card space-y-4">
        <h3 className="text-sm font-semibold text-[#e0e0e0] uppercase tracking-wider flex items-center gap-2">
          <Wrench size={16} className="text-[#4fc3f7]" /> Skills Overview
        </h3>
        <div className="flex flex-wrap gap-2">
          {['code-review', 'refactor', 'test-gen'].map((id) => (
            <span key={id} className={`text-xs font-bold px-2.5 py-1 rounded-full border ${
              id !== 'test-gen' ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' : 'bg-[#2d2d44]/20 text-[#9e9eb0] border-[#2d2d44]/40'
            }`}>{id}</span>
          ))}
        </div>
        <BarChart data={[
          { label: 'Active', value: 2, color: '#34d399' },
          { label: 'Inactive', value: 1, color: '#f87171' },
        ]} title="Skill Activity" />
      </div>
    </div>
  );
}
