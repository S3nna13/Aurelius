import { useState, useEffect } from 'react';
import {
  Brain,
  Search,
  Layers,
  Database,
  Clock,
  Loader2,
  AlertTriangle,
  RefreshCw,
} from 'lucide-react';
import { useApi } from '../hooks/useApi';
import { useToast } from '../components/ToastProvider';

interface MemoryLayer {
  name: string;
  entries: number;
  size: string;
  description: string;
  color: string;
  bg: string;
  border: string;
}

const layerMeta: Record<string, Omit<MemoryLayer, 'entries'>> = {
  'L0 Meta Rules': {
    name: 'Meta Rules',
    size: '2 MB',
    description: 'Core agent behavior rules and constraints.',
    color: 'text-[#4fc3f7]',
    bg: 'bg-[#4fc3f7]/10',
    border: 'border-[#4fc3f7]/20',
  },
  'L1 Insight Index': {
    name: 'Insight Index',
    size: '8 MB',
    description: 'Indexed insights and pattern recognitions.',
    color: 'text-emerald-400',
    bg: 'bg-emerald-500/10',
    border: 'border-emerald-500/20',
  },
  'L2 Global Facts': {
    name: 'Global Facts',
    size: '45 MB',
    description: 'Persistent knowledge and learned patterns.',
    color: 'text-amber-400',
    bg: 'bg-amber-500/10',
    border: 'border-amber-500/20',
  },
  'L3 Task Skills': {
    name: 'Task Skills',
    size: '12 MB',
    description: 'Active task context and skill mappings.',
    color: 'text-rose-400',
    bg: 'bg-rose-500/10',
    border: 'border-rose-500/20',
  },
  'L4 Session Archive': {
    name: 'Session Archive',
    size: '210 MB',
    description: 'Archived conversation sessions and history.',
    color: 'text-purple-400',
    bg: 'bg-purple-500/10',
    border: 'border-purple-500/20',
  },
};

const fallbackLayers = [
  { name: 'Short-term', entries: 1240, size: '12 MB', description: 'Recent conversations and temporary context.', color: 'text-[#4fc3f7]', bg: 'bg-[#4fc3f7]/10', border: 'border-[#4fc3f7]/20' },
  { name: 'Working', entries: 342, size: '45 MB', description: 'Active task context and intermediate results.', color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/20' },
  { name: 'Long-term', entries: 8901, size: '210 MB', description: 'Persistent knowledge and learned patterns.', color: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/20' },
  { name: 'Episodic', entries: 567, size: '34 MB', description: 'Event-based memories with timestamps.', color: 'text-rose-400', bg: 'bg-rose-500/10', border: 'border-rose-500/20' },
];

const recentEntries = [
  { id: 1, layer: 'Short-term', content: 'User asked about system status.', time: '2 min ago' },
  { id: 2, layer: 'Working', content: 'Workflow "Daily Backup" context.', time: '5 min ago' },
  { id: 3, layer: 'Long-term', content: 'Learned pattern: user prefers dark mode.', time: '1 day ago' },
  { id: 4, layer: 'Episodic', content: 'Alert: High CPU on node-2.', time: '18 min ago' },
];

export default function Memory() {
  const [search, setSearch] = useState('');
  const { toast } = useToast();

  const {
    data,
    loading,
    error,
    refresh,
  } = useApi<{ layers: Record<string, number> }>('/memory', {
    refreshInterval: 10000,
  });

  useEffect(() => {
    if (error) {
      toast('Failed to load memory data', 'error');
    }
  }, [error, toast]);

  const memoryLayers: MemoryLayer[] = data?.layers
    ? Object.entries(data.layers).map(([key, count]) => {
        const meta = layerMeta[key];
        if (meta) {
          return { ...meta, name: meta.name, entries: count };
        }
        return {
          name: key,
          entries: count,
          size: '—',
          description: key,
          color: 'text-[#9e9eb0]',
          bg: 'bg-[#2d2d44]/20',
          border: 'border-[#2d2d44]/40',
        };
      })
    : fallbackLayers;

  const filteredEntries = recentEntries.filter((e) =>
    e.content.toLowerCase().includes(search.toLowerCase())
  );

  const totalEntries = memoryLayers.reduce((sum, l) => sum + l.entries, 0);

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
          <Brain size={20} className="text-[#4fc3f7]" />
          Memory Explorer
        </h2>
        <div className="flex items-center gap-3">
          <span className="text-sm text-[#9e9eb0]">
            {totalEntries.toLocaleString()} total entries
          </span>
          <button
            onClick={refresh}
            disabled={loading}
            className="aurelius-btn-outline flex items-center gap-2 text-sm disabled:opacity-50"
          >
            {loading ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
            Refresh
          </button>
        </div>
      </div>

      {error && (
        <div className="aurelius-card border-rose-500/30 bg-rose-500/5 text-rose-300">
          <AlertTriangle size={18} className="inline mr-2" />
          {error.message}
        </div>
      )}

      {/* Search */}
      <div className="relative">
        <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#9e9eb0]" />
        <input
          type="text"
          placeholder="Search memory entries..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full bg-[#0f0f1a] border border-[#2d2d44] rounded-lg pl-9 pr-4 py-2.5 text-sm text-[#e0e0e0] placeholder:text-[#9e9eb0] focus:outline-none focus:border-[#4fc3f7]"
        />
      </div>

      {/* Layers */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {memoryLayers.map((layer) => (
          <div
            key={layer.name}
            className="aurelius-card space-y-3 hover:border-[#4fc3f7]/30 transition-colors"
          >
            <div className="flex items-center gap-2">
              <Layers size={16} className={layer.color} />
              <h3 className="text-sm font-semibold text-[#e0e0e0]">{layer.name}</h3>
            </div>
            <p className="text-xs text-[#9e9eb0]">{layer.description}</p>
            <div className="flex items-center justify-between pt-2 border-t border-[#2d2d44]">
              <div className="flex items-center gap-1 text-xs text-[#9e9eb0]">
                <Database size={12} />
                {layer.entries.toLocaleString()} entries
              </div>
              <span className={`text-xs font-bold px-2 py-0.5 rounded ${layer.bg} ${layer.color} border ${layer.border}`}>
                {layer.size}
              </span>
            </div>
          </div>
        ))}
      </div>

      {/* Recent Entries */}
      <div className="aurelius-card space-y-4">
        <h3 className="text-sm font-semibold text-[#e0e0e0] uppercase tracking-wider flex items-center gap-2">
          <Clock size={16} className="text-[#4fc3f7]" />
          Recent Entries
        </h3>
        <div className="space-y-2">
          {filteredEntries.map((entry) => (
            <div
              key={entry.id}
              className="flex items-center justify-between px-3 py-2.5 rounded-lg bg-[#0f0f1a]/50 border border-[#2d2d44]/50 hover:border-[#4fc3f7]/10 transition-colors"
            >
              <p className="text-sm text-[#e0e0e0]">{entry.content}</p>
              <div className="flex items-center gap-3 shrink-0 ml-4">
                <span className="text-[10px] uppercase font-bold px-1.5 py-0.5 rounded bg-[#2d2d44]/20 text-[#9e9eb0] border border-[#2d2d44]/40">
                  {entry.layer}
                </span>
                <span className="text-xs text-[#9e9eb0]">{entry.time}</span>
              </div>
            </div>
          ))}
          {filteredEntries.length === 0 && (
            <p className="text-sm text-[#9e9eb0] text-center py-6">No entries match your search.</p>
          )}
        </div>
      </div>
    </div>
  );
}
