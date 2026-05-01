import { useState, useRef, useEffect } from 'react';
import { Terminal, Search, Download, Trash2, Filter, Pause, Play } from 'lucide-react';
import Input from '../components/ui/Input';
import Select from '../components/ui/Select';

interface LogEntry { timestamp: string; level: string; module: string; message: string; }

const LEVELS = ['ALL', 'DEBUG', 'INFO', 'WARN', 'ERROR', 'FATAL'];
const LEVEL_COLORS: Record<string, string> = { DEBUG: 'text-gray-400', INFO: 'text-[#4fc3f7]', WARN: 'text-amber-400', ERROR: 'text-rose-400', FATAL: 'text-red-500' };

export default function Logs() {
  const [logs, setLogs] = useState<LogEntry[]>([]);
  const [search, setSearch] = useState('');
  const [levelFilter, setLevelFilter] = useState('ALL');
  const [paused, setPaused] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const es = new EventSource('/api/logs/stream');
    es.onmessage = (e) => {
      if (!paused) {
        try { setLogs(prev => [...prev.slice(-500), JSON.parse(e.data)]); } catch {}
      }
    };
    return () => es.close();
  }, [paused]);

  useEffect(() => { if (!paused) bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [logs, paused]);

  const filtered = logs.filter(l => {
    if (levelFilter !== 'ALL' && l.level !== levelFilter) return false;
    if (search && !l.message.toLowerCase().includes(search.toLowerCase()) && !l.module.toLowerCase().includes(search.toLowerCase())) return false;
    return true;
  });

  const exportLogs = () => {
    const blob = new Blob([filtered.map(l => `[${l.timestamp}] [${l.level}] [${l.module}] ${l.message}`).join('\n')], { type: 'text/plain' });
    const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `logs-${Date.now()}.txt`; a.click();
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2"><Terminal size={20} className="text-[#4fc3f7]" />Logs</h2>
        <div className="flex gap-2">
          <button onClick={() => setPaused(!paused)} className="aurelius-btn-outline p-2">{paused ? <Play size={14} /> : <Pause size={14} />}</button>
          <button onClick={exportLogs} className="aurelius-btn-outline flex items-center gap-1.5 text-xs"><Download size={12} /> Export</button>
          <button onClick={() => setLogs([])} className="aurelius-btn-outline flex items-center gap-1.5 text-xs"><Trash2 size={12} /> Clear</button>
        </div>
      </div>

      <div className="flex gap-2">
        <div className="relative flex-1"><Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#9e9eb0]" /><Input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search logs..." className="pl-8 py-1.5 text-sm w-full" /></div>
        <select value={levelFilter} onChange={e => setLevelFilter(e.target.value)} className="bg-[#0f0f1a] border border-[#2d2d44] rounded-lg px-3 py-1.5 text-sm text-[#e0e0e0] w-24">
          {LEVELS.map(l => <option key={l} value={l}>{l}</option>)}
        </select>
      </div>

      <div className="h-[calc(100vh-16rem)] overflow-y-auto bg-[#0a0a14] rounded-xl border border-[#2d2d44] p-3 font-mono text-xs space-y-0.5">
        {filtered.length === 0 && <p className="text-[#9e9eb0] text-center py-8">No log entries yet.</p>}
        {filtered.map((log, i) => (
          <div key={i} className="flex items-start gap-2 hover:bg-white/[0.02] rounded px-1 py-0.5">
            <span className="text-[#9e9eb0]/50 shrink-0">{log.timestamp}</span>
            <span className={`${LEVEL_COLORS[log.level] || 'text-gray-400'} font-bold shrink-0 w-12`}>{log.level}</span>
            <span className="text-[#4fc3f7]/60 shrink-0">[{log.module}]</span>
            <span className="text-[#9e9eb0] break-all">{log.message}</span>
          </div>
        ))}
        <div ref={bottomRef} />
      </div>
    </div>
  );
}
