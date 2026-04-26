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
  Eye,
  X,
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

interface MemoryEntry {
  id: string;
  content: string;
  layer: string;
  timestamp: string;
  access_count: number;
  importance_score: number;
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

function timeAgo(ts: string): string {
  const d = new Date(ts).getTime();
  const diff = (Date.now() - d) / 1000;
  if (isNaN(diff) || diff < 0) return 'Just now';
  if (diff < 60) return 'Just now';
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} hr ago`;
  return `${Math.floor(diff / 86400)} days ago`;
}

export default function Memory() {
  const [search, setSearch] = useState('');
  const [selectedLayer, setSelectedLayer] = useState<string>('all');
  const [detailEntry, setDetailEntry] = useState<MemoryEntry | null>(null);
  const { toast } = useToast();

  const {
    data,
    loading,
    error,
    refresh,
  } = useApi<{ layers: Record<string, number> }>('/memory', {
    refreshInterval: 10000,
  });

  const {
    data: entriesData,
    loading: entriesLoading,
    error: entriesError,
    refresh: refreshEntries,
  } = useApi<{ entries: MemoryEntry[] }>(
    `/memory/entries?limit=50${search ? `&q=${encodeURIComponent(search)}` : ''}${selectedLayer !== 'all' ? `&layer=${encodeURIComponent(selectedLayer)}` : ''}`,
    { refreshInterval: 10000 }
  );

  useEffect(() => {
    if (error) toast('Failed to load memory data', 'error');
    if (entriesError) toast('Failed to load memory entries', 'error');
  }, [error, entriesError, toast]);

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

  const entries = entriesData?.entries || [];
  const totalEntries = memoryLayers.reduce((sum, l) => sum + l.entries, 0);

  const refreshAll = () => {
    refresh();
    refreshEntries();
  };

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
            onClick={refreshAll}
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
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4">
        <button
          onClick={() => setSelectedLayer('all')}
          className={`aurelius-card space-y-3 text-left transition-colors ${selectedLayer === 'all' ? 'border-aurelius-accent/50' : 'hover:border-[#4fc3f7]/30'}`}
        >
          <div className="flex items-center gap-2">
            <Layers size={16} className="text-aurelius-accent" />
            <h3 className="text-sm font-semibold text-[#e0e0e0]">All Layers</h3>
          </div>
          <p className="text-xs text-[#9e9eb0]">View entries across all memory tiers.</p>
          <div className="flex items-center justify-between pt-2 border-t border-[#2d2d44]">
            <div className="flex items-center gap-1 text-xs text-[#9e9eb0]">
              <Database size={12} />
              {totalEntries.toLocaleString()} entries
            </div>
          </div>
        </button>
        {memoryLayers.map((layer) => (
          <button
            key={layer.name}
            onClick={() => setSelectedLayer(Object.keys(layerMeta).find(k => layerMeta[k].name === layer.name) || layer.name)}
            className={`aurelius-card space-y-3 text-left transition-colors ${selectedLayer === (Object.keys(layerMeta).find(k => layerMeta[k].name === layer.name) || layer.name) ? 'border-aurelius-accent/50' : 'hover:border-[#4fc3f7]/30'}`}
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
          </button>
        ))}
      </div>

      {/* Recent Entries */}
      <div className="aurelius-card space-y-4">
        <h3 className="text-sm font-semibold text-[#e0e0e0] uppercase tracking-wider flex items-center gap-2">
          <Clock size={16} className="text-[#4fc3f7]" />
          Recent Entries
        </h3>
        {entriesLoading && entries.length === 0 && (
          <div className="text-center py-8 text-[#9e9eb0]">
            <Loader2 size={24} className="mx-auto mb-2 animate-spin opacity-60" />
            <p className="text-sm">Loading entries...</p>
          </div>
        )}
        <div className="space-y-2">
          {entries.map((entry) => (
            <div
              key={entry.id}
              onClick={() => setDetailEntry(entry)}
              className="flex items-center justify-between px-3 py-2.5 rounded-lg bg-[#0f0f1a]/50 border border-[#2d2d44]/50 hover:border-[#4fc3f7]/20 transition-colors cursor-pointer"
            >
              <p className="text-sm text-[#e0e0e0] truncate max-w-[60%]">{entry.content}</p>
              <div className="flex items-center gap-3 shrink-0">
                <span className="text-[10px] uppercase font-bold px-1.5 py-0.5 rounded bg-[#2d2d44]/20 text-[#9e9eb0] border border-[#2d2d44]/40">
                  {entry.layer}
                </span>
                <span className="text-xs text-[#9e9eb0]">{timeAgo(entry.timestamp)}</span>
                <Eye size={14} className="text-aurelius-muted opacity-0 group-hover:opacity-100" />
              </div>
            </div>
          ))}
          {entries.length === 0 && !entriesLoading && (
            <p className="text-sm text-[#9e9eb0] text-center py-6">No entries found.</p>
          )}
        </div>
      </div>

      {/* Detail Modal */}
      {detailEntry && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
          <div className="bg-aurelius-card border border-aurelius-border rounded-xl p-6 max-w-lg w-full shadow-2xl">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-bold text-aurelius-text">Memory Entry</h3>
              <button
                onClick={() => setDetailEntry(null)}
                className="p-1 rounded-lg text-aurelius-muted hover:text-aurelius-text hover:bg-aurelius-border/40 transition-colors"
              >
                <X size={18} />
              </button>
            </div>
            <div className="space-y-3">
              <div>
                <p className="text-xs text-aurelius-muted uppercase tracking-wider mb-1">Content</p>
                <p className="text-sm text-aurelius-text leading-relaxed">{detailEntry.content}</p>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div className="bg-aurelius-bg border border-aurelius-border rounded-lg p-3">
                  <p className="text-xs text-aurelius-muted uppercase tracking-wider">Layer</p>
                  <p className="text-sm font-medium text-aurelius-text mt-0.5">{detailEntry.layer}</p>
                </div>
                <div className="bg-aurelius-bg border border-aurelius-border rounded-lg p-3">
                  <p className="text-xs text-aurelius-muted uppercase tracking-wider">Access Count</p>
                  <p className="text-sm font-medium text-aurelius-text mt-0.5">{detailEntry.access_count}</p>
                </div>
                <div className="bg-aurelius-bg border border-aurelius-border rounded-lg p-3">
                  <p className="text-xs text-aurelius-muted uppercase tracking-wider">Importance</p>
                  <p className="text-sm font-medium text-aurelius-text mt-0.5">{(detailEntry.importance_score * 100).toFixed(0)}%</p>
                </div>
                <div className="bg-aurelius-bg border border-aurelius-border rounded-lg p-3">
                  <p className="text-xs text-aurelius-muted uppercase tracking-wider">Timestamp</p>
                  <p className="text-sm font-medium text-aurelius-text mt-0.5">{new Date(detailEntry.timestamp).toLocaleString()}</p>
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
