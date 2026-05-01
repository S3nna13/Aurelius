import { useState } from 'react';
import { motion } from 'framer-motion';
import { Bot, Search, Filter, Star, Clock, Code, BookOpen, Server, MessageSquare, Pen, GraduationCap, Database, Calendar, Wrench, Shield, BarChart3 } from 'lucide-react';
import Input from '../components/ui/Input';
import { useApi } from '../hooks/useApi';
import EmptyState from '../components/EmptyState';
import Skeleton from '../components/Skeleton';

const CATEGORY_ICONS: Record<string, typeof Bot> = {
  coding: Code, research: BookOpen, devops: Server, communication: MessageSquare,
  creative: Pen, education: GraduationCap, data: Database, productivity: Calendar, meta: Wrench,
};

const CATEGORY_COLORS: Record<string, string> = {
  coding: 'text-emerald-400', research: 'text-[#4fc3f7]', devops: 'text-amber-400',
  communication: 'text-violet-400', creative: 'text-rose-400', education: 'text-cyan-400',
  data: 'text-indigo-400', productivity: 'text-lime-400', meta: 'text-gray-400',
};

interface RegistryAgent {
  id: string; name: string; category: string; description: string; capabilities: string[]; enabled: boolean;
}

export default function AgentComparison() {
  const [search, setSearch] = useState('');
  const [category, setCategory] = useState('all');
  const { data, loading } = useApi<{ agents: RegistryAgent[]; categories: string[] }>('/registry', { refreshInterval: 5000 });
  const agents = data?.agents || [];
  const categories = data?.categories || [];

  const filtered = agents.filter(a => {
    if (category !== 'all' && a.category !== category) return false;
    return !search || a.name.toLowerCase().includes(search.toLowerCase()) || a.description.toLowerCase().includes(search.toLowerCase());
  });

  const enabled = agents.filter(a => a.enabled).length;
  const categories_list = ['all', ...categories];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
          <Bot size={20} className="text-[#4fc3f7]" /> Agent Registry
          <span className="text-sm font-normal text-[#9e9eb0]">({agents.length} types, {enabled} active)</span>
        </h2>
        <div className="flex gap-2">
          <div className="relative"><Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#9e9eb0]" /><Input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search agents..." className="pl-8 py-1.5 text-sm w-48" /></div>
          <select value={category} onChange={e => setCategory(e.target.value)} className="bg-[#0f0f1a] border border-[#2d2d44] rounded-lg px-3 py-1.5 text-sm text-[#e0e0e0]">
            {categories_list.map(c => <option key={c} value={c}>{c === 'all' ? 'All Types' : c.charAt(0).toUpperCase() + c.slice(1)}</option>)}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 lg:grid-cols-7 gap-2">
        {categories.map(c => {
          const Icon = CATEGORY_ICONS[c] || Bot;
          const count = agents.filter(a => a.category === c).length;
          const active = agents.filter(a => a.category === c && a.enabled).length;
          return (
            <button key={c} onClick={() => setCategory(c)}
              className={`aurelius-card p-3 text-center hover:border-[#4fc3f7]/30 transition-all ${category === c ? 'border-[#4fc3f7]/50' : ''}`}>
              <Icon size={16} className={`mx-auto mb-1 ${CATEGORY_COLORS[c] || 'text-[#4fc3f7]'}`} />
              <p className="text-[10px] text-[#e0e0e0] font-medium">{c}</p>
              <p className="text-[9px] text-[#9e9eb0]">{active}/{count} active</p>
            </button>
          );
        })}
      </div>

      {loading && <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">{ [1,2,3,4,5,6].map(i => <Skeleton key={i} className="h-28" />) }</div>}
      {!loading && filtered.length === 0 && <EmptyState icon={Bot} title="No Agent Types" description="All agent types are registered. Try adjusting filters." />}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {filtered.map(agent => {
          const Icon = CATEGORY_ICONS[agent.category] || Bot;
          const color = CATEGORY_COLORS[agent.category] || 'text-[#4fc3f7]';
          return (
            <motion.div key={agent.id} layout initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              className="aurelius-card p-4 hover:border-[#4fc3f7]/30 transition-all group"
            >
              <div className="flex items-start gap-3 mb-3">
                <div className={`w-9 h-9 rounded-lg flex items-center justify-center ${color} bg-white/5`}>
                  <Icon size={16} />
                </div>
                <div className="min-w-0 flex-1">
                  <h3 className="text-sm font-semibold text-[#e0e0e0] truncate">{agent.name}</h3>
                  <div className="flex items-center gap-2">
                    <span className={`text-[10px] ${color}`}>{agent.category}</span>
                    {agent.enabled && <span className="text-[10px] text-emerald-400">● Active</span>}
                  </div>
                </div>
              </div>
              <p className="text-xs text-[#9e9eb0] line-clamp-2 mb-3">{agent.description}</p>
              <div className="flex flex-wrap gap-1">
                {agent.capabilities.slice(0, 5).map(cap => (
                  <span key={cap} className="text-[9px] text-[#4fc3f7] bg-[#4fc3f7]/10 px-1.5 py-0.5 rounded">{cap}</span>
                ))}
              </div>
            </motion.div>
          );
        })}
      </div>

      <div className="aurelius-card p-4">
        <h3 className="text-sm font-semibold text-[#e0e0e0] mb-3">Registry Summary</h3>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3 text-center text-xs">
          {categories.map(c => {
            const count = agents.filter(a => a.category === c).length;
            const active = agents.filter(a => a.category === c && a.enabled).length;
            const Icon = CATEGORY_ICONS[c] || Bot;
            const color = CATEGORY_COLORS[c] || 'text-[#4fc3f7]';
            return (
              <div key={c} className="bg-[#0f0f1a] p-2 rounded-lg">
                <Icon size={12} className={`mx-auto ${color}`} />
                <p className={`font-bold mt-0.5 ${color}`}>{active}/{count}</p>
                <p className="text-[9px] text-[#9e9eb0]">{c}</p>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
