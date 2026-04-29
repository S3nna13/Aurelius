import { useState } from 'react';
import { motion } from 'framer-motion';
import { Bot, Search, Filter, Star, Clock, Users, Code, BookOpen, Globe, Cpu, Sparkles } from 'lucide-react';
import Input from '../components/ui/Input';
import { useApi } from '../hooks/useApi';
import EmptyState from '../components/EmptyState';
import Skeleton from '../components/Skeleton';

interface AgentTemplate {
  id: string; name: string; description: string;
  category: string; capabilities: string[];
  popularity: number; agents: number;
  icon: typeof Bot;
}

const DEFAULT_TEMPLATES: AgentTemplate[] = [
  { id: 'code-reviewer', name: 'Code Reviewer', description: 'Reviews pull requests for bugs, style issues, and security vulnerabilities.', category: 'coding', capabilities: ['code', 'review', 'security'], popularity: 95, agents: 12, icon: Code },
  { id: 'research-assistant', name: 'Research Assistant', description: 'Searches the web, summarizes findings, and generates reports.', category: 'research', capabilities: ['search', 'summarize', 'analyze'], popularity: 88, agents: 8, icon: BookOpen },
  { id: 'devops-bot', name: 'DevOps Bot', description: 'Monitors infrastructure, deploys services, and manages incidents.', category: 'devops', capabilities: ['monitor', 'deploy', 'incident'], popularity: 82, agents: 6, icon: Globe },
  { id: 'data-analyst', name: 'Data Analyst', description: 'Analyzes datasets, generates visualizations, and produces insights.', category: 'analysis', capabilities: ['analyze', 'visualize', 'report'], popularity: 79, agents: 5, icon: Cpu },
  { id: 'tutor', name: 'AI Tutor', description: 'Teaches concepts through Socratic dialogue and adaptive exercises.', category: 'education', capabilities: ['teach', 'explain', 'assess'], popularity: 74, agents: 4, icon: Users },
  { id: 'creative-writer', name: 'Creative Writer', description: 'Generates stories, poems, marketing copy, and creative content.', category: 'creative', capabilities: ['write', 'edit', 'brainstorm'], popularity: 71, agents: 3, icon: Star },
];

const CATEGORIES = ['all', 'coding', 'research', 'devops', 'analysis', 'education', 'creative'];

export default function AgentTemplates() {
  const [search, setSearch] = useState('');
  const [category, setCategory] = useState('all');
  const [selected, setSelected] = useState<AgentTemplate | null>(null);

  const templates = DEFAULT_TEMPLATES.filter(t => {
    if (category !== 'all' && t.category !== category) return false;
    return !search || t.name.toLowerCase().includes(search.toLowerCase()) || t.description.toLowerCase().includes(search.toLowerCase());
  });

  const handleDeploy = async (template: AgentTemplate) => {
    try {
      const res = await fetch('/api/agents', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          name: template.name.replace(/\s+/g, '-').toLowerCase(),
          capabilities: template.capabilities,
          role: 'worker',
        }),
      });
      if (res.ok) setSelected(null);
    } catch {}
  };

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
        <Sparkles size={20} className="text-[#4fc3f7]" /> Agent Templates
      </h2>

      <div className="flex gap-2">
        <div className="relative flex-1 max-w-xs">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#9e9eb0]" />
          <Input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search templates..." className="pl-8 py-1.5 text-sm w-full" />
        </div>
        <select value={category} onChange={e => setCategory(e.target.value)}
          className="bg-[#0f0f1a] border border-[#2d2d44] rounded-lg px-3 py-1.5 text-sm text-[#e0e0e0]">
          {CATEGORIES.map(c => <option key={c} value={c}>{c === 'all' ? 'All Categories' : c}</option>)}
        </select>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {templates.map(t => {
          const Icon = t.icon;
          return (
            <motion.div key={t.id} layout initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              onClick={() => setSelected(t)}
              className="aurelius-card p-4 hover:border-[#4fc3f7]/30 cursor-pointer transition-all group"
            >
              <div className="flex items-center gap-3 mb-3">
                <div className="w-10 h-10 rounded-lg bg-[#4fc3f7]/10 flex items-center justify-center">
                  <Icon size={20} className="text-[#4fc3f7]" />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-[#e0e0e0]">{t.name}</h3>
                  <div className="flex items-center gap-2 text-[10px] text-[#9e9eb0]">
                    <span>{t.category}</span>
                    <span>·</span>
                    <span>{t.agents} deployed</span>
                  </div>
                </div>
              </div>
              <p className="text-xs text-[#9e9eb0] line-clamp-2 mb-3">{t.description}</p>
              <div className="flex items-center justify-between">
                <div className="flex gap-1">
                  {t.capabilities.slice(0, 3).map(cap => (
                    <span key={cap} className="text-[10px] text-[#4fc3f7] bg-[#4fc3f7]/10 px-2 py-0.5 rounded-full">{cap}</span>
                  ))}
                </div>
                <span className="text-[10px] text-amber-400 flex items-center gap-1">
                  <Star size={10} /> {t.popularity}%
                </span>
              </div>
            </motion.div>
          );
        })}
      </div>

      {selected && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4" onClick={() => setSelected(null)}>
          <div className="aurelius-card p-6 max-w-md w-full space-y-4" onClick={e => e.stopPropagation()}>
            <div className="flex items-center gap-3">
              <div className="w-12 h-12 rounded-xl bg-[#4fc3f7]/10 flex items-center justify-center">
                <Bot size={24} className="text-[#4fc3f7]" />
              </div>
              <div>
                <h3 className="text-lg font-bold text-[#e0e0e0]">{selected.name}</h3>
                <p className="text-xs text-[#9e9eb0]">{selected.category} · {selected.agents} active deployments</p>
              </div>
            </div>
            <p className="text-sm text-[#9e9eb0]">{selected.description}</p>
            <div className="flex flex-wrap gap-1.5">
              {selected.capabilities.map(cap => (
                <span key={cap} className="text-[10px] text-[#4fc3f7] bg-[#4fc3f7]/10 px-2 py-0.5 rounded-full">{cap}</span>
              ))}
            </div>
            <button onClick={() => handleDeploy(selected)} className="aurelius-btn-primary w-full flex items-center justify-center gap-2">
              <Bot size={14} /> Deploy Agent
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
