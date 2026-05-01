import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  BarChart3, TrendingUp, Clock, Activity, Bot,
  MessageSquare, Cpu, Database, Download, Calendar,
} from 'lucide-react';
import { useApi } from '../hooks/useApi';
import StatsCard from '../components/StatsCard';
import BarChart from '../components/charts/BarChart';
import AreaChart from '../components/charts/AreaChart';
import DonutChart from '../components/charts/DonutChart';
import Select from '../components/ui/Select';

interface AnalyticsData {
  requests: { total: number; byModel: Record<string, number>; byEndpoint: Record<string, number> };
  latency: { avg: number; p50: number; p95: number; p99: number; history: Array<{ time: string; value: number }> };
  agents: { total: number; active: number; tasksCompleted: number; errorRate: number };
  tokens: { total: number; byDay: Array<{ date: string; tokens: number }> };
  errors: { total: number; byType: Record<string, number> };
}

export default function Analytics() {
  const [timeRange, setTimeRange] = useState('24h');
  const { data, loading } = useApi<AnalyticsData>(`/stats?range=${timeRange}`, { refreshInterval: 10000 });

  if (loading && !data) {
    return (
      <div className="space-y-4">
        <div className="h-8 w-48 bg-[#1a1a2e] rounded animate-pulse" />
        <div className="grid grid-cols-4 gap-3">{[1,2,3,4].map(i => <div key={i} className="h-24 bg-[#1a1a2e] rounded animate-pulse" />)}</div>
        <div className="h-64 bg-[#1a1a2e] rounded animate-pulse" />
      </div>
    );
  }

  const stats = data || {
    requests: { total: 0, byModel: {}, byEndpoint: {} },
    latency: { avg: 0, p50: 0, p95: 0, p99: 0, history: [] },
    agents: { total: 0, active: 0, tasksCompleted: 0, errorRate: 0 },
    tokens: { total: 0, byDay: [] },
    errors: { total: 0, byType: {} },
  };

  const modelData = Object.entries(stats.requests.byModel).map(([name, value]) => ({ name, value }));
  const endpointData = Object.entries(stats.requests.byEndpoint).map(([name, value]) => ({ name, value }));
  const tokenHistory = stats.tokens.byDay.map(d => ({ label: d.date, value: d.tokens }));
  const latencyHistory = stats.latency.history.map(p => ({ label: p.time, value: p.value }));

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
          <BarChart3 size={20} className="text-[#4fc3f7]" />
          Analytics
        </h2>
        <div className="flex gap-2">
          <select value={timeRange} onChange={e => setTimeRange(e.target.value)}
            className="bg-[#0f0f1a] border border-[#2d2d44] rounded-lg px-3 py-1.5 text-sm text-[#e0e0e0]">
            <option value="1h">Last hour</option>
            <option value="24h">Last 24h</option>
            <option value="7d">Last 7 days</option>
            <option value="30d">Last 30 days</option>
          </select>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatsCard label="Total Requests" value={stats.requests.total.toLocaleString()} icon={Activity} color="text-[#4fc3f7]" />
        <StatsCard label="Avg Latency" value={`${stats.latency.avg.toFixed(0)}ms`} icon={Clock} color="text-emerald-400" />
        <StatsCard label="Active Agents" value={stats.agents.active} icon={Bot} color="text-amber-400" />
        <StatsCard label="Tokens Used" value={(stats.tokens.total / 1000).toFixed(0) + 'K'} icon={Database} color="text-violet-400" />
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="aurelius-card p-4">
          <h3 className="text-sm font-semibold text-[#e0e0e0] mb-3 flex items-center gap-2">
            <BarChart3 size={14} className="text-[#4fc3f7]" /> Requests by Model
          </h3>
          <div className="h-48">
            <BarChart data={modelData} xKey="name" yKey="value" color="#4fc3f7" />
          </div>
        </div>

        <div className="aurelius-card p-4">
          <h3 className="text-sm font-semibold text-[#e0e0e0] mb-3 flex items-center gap-2">
            <Clock size={14} className="text-amber-400" /> Latency (p50/p95/p99)
          </h3>
          <div className="text-center py-8">
            <div className="grid grid-cols-3 gap-4">
              <div><p className="text-2xl font-bold text-emerald-400">{stats.latency.p50.toFixed(0)}</p><p className="text-xs text-[#9e9eb0]">p50 ms</p></div>
              <div><p className="text-2xl font-bold text-amber-400">{stats.latency.p95.toFixed(0)}</p><p className="text-xs text-[#9e9eb0]">p95 ms</p></div>
              <div><p className="text-2xl font-bold text-rose-400">{stats.latency.p99.toFixed(0)}</p><p className="text-xs text-[#9e9eb0]">p99 ms</p></div>
            </div>
          </div>
          {latencyHistory.length > 0 && (
            <div className="h-32"><AreaChart data={latencyHistory} xKey="label" yKey="value" color="#4fc3f7" /></div>
          )}
        </div>

        <div className="aurelius-card p-4">
          <h3 className="text-sm font-semibold text-[#e0e0e0] mb-3 flex items-center gap-2">
            <Database size={14} className="text-violet-400" /> Token Usage
          </h3>
          <div className="h-48">
            <AreaChart data={tokenHistory} xKey="label" yKey="value" color="#a78bfa" />
          </div>
        </div>

        <div className="aurelius-card p-4">
          <h3 className="text-sm font-semibold text-[#e0e0e0] mb-3 flex items-center gap-2">
            <Activity size={14} className="text-rose-400" /> Endpoint Distribution
          </h3>
          <div className="h-48">
            <DonutChart data={endpointData} />
          </div>
        </div>
      </div>
    </div>
  );
}
