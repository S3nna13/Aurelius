import { useState } from 'react';
import { motion } from 'framer-motion';
import { Database, Search, Star, Clock, RefreshCw, Plus, Trash2 } from 'lucide-react';
import Input from '../components/ui/Input';
import { useApi } from '../hooks/useApi';
import EmptyState from '../components/EmptyState';

interface MemoryEntry { id: string; content: string; type: string; layer: string; timestamp: number; tags: string[]; }

export default function Memory() {
  const [search, setSearch] = useState('');
  const [layer, setLayer] = useState('all');
  const { data, loading } = useApi<{ entries: MemoryEntry[] }>('/memory', { refreshInterval: 10000 });
  const entries = (data?.entries || []).filter(e => {
    if (layer !== 'all' && e.layer !== layer) return false;
    return !search || e.content.toLowerCase().includes(search.toLowerCase()) || e.tags?.some(t => t.includes(search.toLowerCase()));
  });

  const layers = ['all', ...new Set((data?.entries || []).map(e => e.layer))];

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2"><Database size={20} className="text-[#4fc3f7]" />Memory</h2>
        <div className="flex gap-2">
          <select value={layer} onChange={e => setLayer(e.target.value)} className="bg-[#0f0f1a] border border-[#2d2d44] rounded-lg px-3 py-1.5 text-sm text-[#e0e0e0]">
            {layers.map(l => <option key={l} value={l}>{l === 'all' ? 'All Layers' : l}</option>)}
          </select>
          <div className="relative"><Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#9e9eb0]" /><Input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search memory..." className="pl-8 py-1.5 text-sm w-48" /></div>
        </div>
      </div>
      {entries.length === 0 && !loading && <EmptyState icon={Database} title="No Memory Entries" description="Agent interactions will be stored here." />}
      <div className="space-y-2">
        {entries.map(entry => (
          <motion.div key={entry.id} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="aurelius-card p-4">
            <div className="flex items-center gap-2 mb-2">
              <span className="text-[10px] font-bold text-[#4fc3f7] bg-[#4fc3f7]/10 px-2 py-0.5 rounded-full">{entry.type}</span>
              <span className="text-[10px] text-[#9e9eb0] bg-[#0f0f1a] px-2 py-0.5 rounded-full">{entry.layer}</span>
              <span className="text-[10px] text-[#9e9eb0]">{new Date(entry.timestamp).toLocaleString()}</span>
            </div>
            <p className="text-sm text-[#e0e0e0]">{entry.content}</p>
            {entry.tags?.length > 0 && (
              <div className="flex gap-1 mt-2">{entry.tags.map(tag => <span key={tag} className="text-[10px] text-[#9e9eb0] bg-[#0f0f1a] px-2 py-0.5 rounded-full">{tag}</span>)}</div>
            )}
          </motion.div>
        ))}
      </div>
    </div>
  );
}
