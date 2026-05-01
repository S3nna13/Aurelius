import { useState } from 'react';
import { motion } from 'framer-motion';
import { Bot, Download, Trash2, RefreshCw, Cpu, Database, CheckCircle, Globe, Server } from 'lucide-react';
import { useApi } from '../hooks/useApi';
import StatsCard from '../components/StatsCard';
import EmptyState from '../components/EmptyState';

interface ModelInfo { id: string; name?: string; provider: string; loaded: boolean; max_context: number; size?: string; parameters?: string; }

export default function Models() {
  const [filter, setFilter] = useState('all');
  const { data, loading } = useApi<{ models: ModelInfo[] }>('/chat/models', { refreshInterval: 10000 });
  const models = data?.models || [];

  const filtered = filter === 'all' ? models : models.filter(m => m.provider === filter);
  const providers = [...new Set(models.map(m => m.provider))];

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2"><Bot size={20} className="text-[#4fc3f7]" />Model Hub</h2>

      <div className="grid grid-cols-2 sm:grid-cols-5 gap-3">
        <StatsCard label="Total Models" value={models.length} icon={Database} color="text-[#4fc3f7]" />
        <StatsCard label="Aurelius" value={models.filter(m => m.provider === 'aurelius').length} icon={Server} color="text-emerald-400" />
        <StatsCard label="OpenAI" value={models.filter(m => m.provider === 'openai').length} icon={Globe} color="text-amber-400" />
        <StatsCard label="Loaded" value={models.filter(m => m.loaded).length} icon={Cpu} color="text-violet-400" />
        <StatsCard label="Max Context" value={`${Math.max(...models.map(m => m.max_context))}`} icon={CheckCircle} color="text-cyan-400" />
      </div>

      <div className="flex gap-2">
        <button onClick={() => setFilter('all')} className={`text-xs px-3 py-1.5 rounded-lg ${filter === 'all' ? 'bg-[#4fc3f7]/10 text-[#4fc3f7] border border-[#4fc3f7]/20' : 'text-[#9e9eb0] hover:text-[#e0e0e0]'}`}>All</button>
        {providers.map(p => (
          <button key={p} onClick={() => setFilter(p)} className={`text-xs px-3 py-1.5 rounded-lg capitalize ${filter === p ? 'bg-[#4fc3f7]/10 text-[#4fc3f7] border border-[#4fc3f7]/20' : 'text-[#9e9eb0] hover:text-[#e0e0e0]'}`}>{p}</button>
        ))}
      </div>

      {filtered.length === 0 && !loading && <EmptyState icon={Bot} title="No Models" description="Models will appear here when providers are configured." />}

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {filtered.map(model => (
          <motion.div key={model.id} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="aurelius-card p-4">
            <div className="flex items-center gap-3 mb-3">
              <div className={`w-9 h-9 rounded-lg flex items-center justify-center ${model.provider === 'aurelius' ? 'bg-emerald-500/10' : 'bg-amber-500/10'}`}>
                {model.provider === 'aurelius' ? <Server size={16} className="text-emerald-400" /> : <Globe size={16} className="text-amber-400" />}
              </div>
              <div className="min-w-0 flex-1">
                <h3 className="text-sm font-medium text-[#e0e0e0] truncate">{model.id}</h3>
                <div className="flex items-center gap-2 text-[10px] text-[#9e9eb0]">
                  <span className={`capitalize ${model.provider === 'aurelius' ? 'text-emerald-400' : 'text-amber-400'}`}>{model.provider}</span>
                  <span>·</span>
                  <span>{model.max_context.toLocaleString()} ctx</span>
                </div>
              </div>
              {model.loaded && <CheckCircle size={14} className="text-emerald-400 shrink-0" />}
            </div>
            <div className="flex gap-2">
              <span className={`text-[10px] px-2 py-0.5 rounded-full ${model.loaded ? 'text-emerald-400 bg-emerald-500/10' : 'text-[#9e9eb0] bg-[#0f0f1a]'}`}>
                {model.loaded ? 'Loaded' : 'Available'}
              </span>
              <span className="text-[10px] text-[#9e9eb0] bg-[#0f0f1a] px-2 py-0.5 rounded-full capitalize">{model.provider}</span>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
