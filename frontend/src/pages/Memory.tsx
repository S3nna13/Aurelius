import { Brain, Search, Layers, Database, Clock } from 'lucide-react';
import { useState } from 'react';

const memoryLayers = [
  {
    name: 'Short-term',
    entries: 1240,
    size: '12 MB',
    description: 'Recent conversations and temporary context.',
    color: 'text-aurelius-accent',
    bg: 'bg-aurelius-accent/10',
    border: 'border-aurelius-accent/20',
  },
  {
    name: 'Working',
    entries: 342,
    size: '45 MB',
    description: 'Active task context and intermediate results.',
    color: 'text-emerald-400',
    bg: 'bg-emerald-500/10',
    border: 'border-emerald-500/20',
  },
  {
    name: 'Long-term',
    entries: 8901,
    size: '210 MB',
    description: 'Persistent knowledge and learned patterns.',
    color: 'text-amber-400',
    bg: 'bg-amber-500/10',
    border: 'border-amber-500/20',
  },
  {
    name: 'Episodic',
    entries: 567,
    size: '34 MB',
    description: 'Event-based memories with timestamps.',
    color: 'text-rose-400',
    bg: 'bg-rose-500/10',
    border: 'border-rose-500/20',
  },
];

const recentEntries = [
  { id: 1, layer: 'Short-term', content: 'User asked about system status.', time: '2 min ago' },
  { id: 2, layer: 'Working', content: 'Workflow "Daily Backup" context.', time: '5 min ago' },
  { id: 3, layer: 'Long-term', content: 'Learned pattern: user prefers dark mode.', time: '1 day ago' },
  { id: 4, layer: 'Episodic', content: 'Alert: High CPU on node-2.', time: '18 min ago' },
];

export default function Memory() {
  const [search, setSearch] = useState('');

  const filteredEntries = recentEntries.filter((e) =>
    e.content.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-aurelius-text flex items-center gap-2">
          <Brain size={20} className="text-aurelius-accent" />
          Memory Explorer
        </h2>
      </div>

      {/* Search */}
      <div className="relative">
        <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-aurelius-muted" />
        <input
          type="text"
          placeholder="Search memory entries..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          className="w-full bg-aurelius-bg border border-aurelius-border rounded-lg pl-9 pr-4 py-2.5 text-sm text-aurelius-text placeholder:text-aurelius-muted focus:outline-none focus:border-aurelius-accent"
        />
      </div>

      {/* Layers */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {memoryLayers.map((layer) => (
          <div
            key={layer.name}
            className="aurelius-card space-y-3 hover:border-aurelius-accent/30 transition-colors"
          >
            <div className="flex items-center gap-2">
              <Layers size={16} className={layer.color} />
              <h3 className="text-sm font-semibold text-aurelius-text">{layer.name}</h3>
            </div>
            <p className="text-xs text-aurelius-muted">{layer.description}</p>
            <div className="flex items-center justify-between pt-2 border-t border-aurelius-border">
              <div className="flex items-center gap-1 text-xs text-aurelius-muted">
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
        <h3 className="text-sm font-semibold text-aurelius-text uppercase tracking-wider flex items-center gap-2">
          <Clock size={16} className="text-aurelius-accent" />
          Recent Entries
        </h3>
        <div className="space-y-2">
          {filteredEntries.map((entry) => (
            <div
              key={entry.id}
              className="flex items-center justify-between px-3 py-2.5 rounded-lg bg-aurelius-bg/50 border border-aurelius-border/50 hover:border-aurelius-accent/10 transition-colors"
            >
              <p className="text-sm text-aurelius-text">{entry.content}</p>
              <div className="flex items-center gap-3 shrink-0 ml-4">
                <span className="text-[10px] uppercase font-bold px-1.5 py-0.5 rounded bg-aurelius-border/20 text-aurelius-muted border border-aurelius-border/40">
                  {entry.layer}
                </span>
                <span className="text-xs text-aurelius-muted">{entry.time}</span>
              </div>
            </div>
          ))}
          {filteredEntries.length === 0 && (
            <p className="text-sm text-aurelius-muted text-center py-6">No entries match your search.</p>
          )}
        </div>
      </div>
    </div>
  );
}
