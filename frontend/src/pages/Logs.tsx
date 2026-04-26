import { useState, useEffect, useMemo } from 'react';
import {
  ScrollText,
  Search,
  Loader2,
  AlertTriangle,
  RefreshCw,
  Filter,
} from 'lucide-react';
import { useApi } from '../hooks/useApi';
import { useToast } from '../components/ToastProvider';
import Pagination from '../components/Pagination';

interface LogEntry {
  timestamp: string;
  level: string;
  logger: string;
  message: string;
}

const levels = ['ALL', 'DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'];

const levelColors: Record<string, string> = {
  DEBUG: 'text-[#9e9eb0]',
  INFO: 'text-[#4fc3f7]',
  WARNING: 'text-amber-400',
  ERROR: 'text-rose-400',
  CRITICAL: 'text-rose-500 font-bold',
};

const levelBg: Record<string, string> = {
  DEBUG: 'bg-[#2d2d44]/20',
  INFO: 'bg-[#4fc3f7]/5',
  WARNING: 'bg-amber-500/5',
  ERROR: 'bg-rose-500/5',
  CRITICAL: 'bg-rose-500/10',
};

export default function Logs() {
  const [search, setSearch] = useState('');
  const [level, setLevel] = useState('ALL');
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 15;
  const { toast } = useToast();

  const query = `/logs?limit=200${level !== 'ALL' ? `&level=${encodeURIComponent(level)}` : ''}${search ? `&q=${encodeURIComponent(search)}` : ''}`;

  const {
    data,
    loading,
    error,
    refresh,
  } = useApi<{ entries: LogEntry[] }>(query, {
    refreshInterval: 5000,
  });

  useEffect(() => {
    if (error) toast('Failed to load logs', 'error');
  }, [error, toast]);

  const entries = data?.entries || [];

  const filtered = useMemo(() => {
    let result = [...entries];
    if (level !== 'ALL') {
      result = result.filter((e) => e.level === level);
    }
    if (search) {
      const sq = search.toLowerCase();
      result = result.filter((e) => e.message.toLowerCase().includes(sq) || e.logger.toLowerCase().includes(sq));
    }
    return result;
  }, [entries, level, search]);

  const totalPages = Math.max(1, Math.ceil(filtered.length / itemsPerPage));
  const paginated = filtered.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage);

  useEffect(() => {
    setCurrentPage(1);
  }, [level, search]);

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
          <ScrollText size={20} className="text-[#4fc3f7]" />
          System Logs
        </h2>
        <button
          onClick={refresh}
          disabled={loading}
          className="aurelius-btn-outline flex items-center gap-2 text-sm disabled:opacity-50 self-start sm:self-auto"
        >
          {loading ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
          Refresh
        </button>
      </div>

      {error && (
        <div className="aurelius-card border-rose-500/30 bg-rose-500/5 text-rose-300">
          <AlertTriangle size={18} className="inline mr-2" />
          {error.message}
        </div>
      )}

      {/* Filters */}
      <div className="flex flex-col sm:flex-row gap-3">
        <div className="relative flex-1">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#9e9eb0]" />
          <input
            type="text"
            placeholder="Search logs..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="w-full bg-[#0f0f1a] border border-[#2d2d44] rounded-lg pl-9 pr-4 py-2.5 text-sm text-[#e0e0e0] placeholder:text-[#9e9eb0] focus:outline-none focus:border-[#4fc3f7]"
          />
        </div>
        <div className="flex items-center gap-2">
          <Filter size={14} className="text-[#9e9eb0]" />
          <select
            value={level}
            onChange={(e) => setLevel(e.target.value)}
            className="bg-[#0f0f1a] border border-[#2d2d44] rounded-lg px-3 py-2.5 text-sm text-[#e0e0e0] focus:outline-none focus:border-[#4fc3f7]"
          >
            {levels.map((l) => (
              <option key={l} value={l}>{l}</option>
            ))}
          </select>
        </div>
      </div>

      {/* Stats */}
      <div className="flex gap-3 flex-wrap">
        {['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'].map((l) => {
          const count = entries.filter((e) => e.level === l).length;
          return (
            <button
              key={l}
              onClick={() => setLevel(level === l ? 'ALL' : l)}
              className={`flex items-center gap-1.5 px-2.5 py-1 rounded-lg text-xs font-medium border transition-colors ${
                level === l
                  ? `${levelBg[l]} ${levelColors[l]} border-current`
                  : 'bg-aurelius-bg text-aurelius-muted border-aurelius-border hover:border-aurelius-accent/30'
              }`}
            >
              <span className={`w-1.5 h-1.5 rounded-full ${levelColors[l].split(' ')[0].replace('text-', 'bg-')}`} />
              {l}: {count}
            </button>
          );
        })}
      </div>

      {/* Log Table */}
      <div className="aurelius-card overflow-hidden p-0">
        {loading && entries.length === 0 && (
          <div className="text-center py-12 text-[#9e9eb0]">
            <Loader2 size={24} className="mx-auto mb-2 animate-spin opacity-60" />
            <p className="text-sm">Loading logs...</p>
          </div>
        )}
        <div className="divide-y divide-aurelius-border/50">
          {paginated.map((entry, i) => (
            <div
              key={i}
              className={`px-4 py-3 flex items-start gap-3 ${levelBg[entry.level] || ''}`}
            >
              <span className={`text-[10px] font-bold uppercase tracking-wider w-16 shrink-0 mt-0.5 ${levelColors[entry.level] || 'text-[#9e9eb0]'}`}>
                {entry.level}
              </span>
              <div className="flex-1 min-w-0">
                <p className="text-sm text-[#e0e0e0] font-mono break-words">{entry.message}</p>
                <p className="text-[10px] text-[#9e9eb0] mt-0.5">
                  {entry.timestamp} · {entry.logger}
                </p>
              </div>
            </div>
          ))}
        </div>
        {paginated.length === 0 && !loading && (
          <div className="text-center py-12 text-[#9e9eb0]">
            <ScrollText size={24} className="mx-auto mb-2 opacity-40" />
            <p className="text-sm">No logs match your filters.</p>
          </div>
        )}
      </div>

      {filtered.length > itemsPerPage && (
        <Pagination currentPage={currentPage} totalPages={totalPages} onPageChange={setCurrentPage} />
      )}
    </div>
  );
}
