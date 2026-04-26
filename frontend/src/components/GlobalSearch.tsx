import { useState, useEffect, useRef } from 'react';
import { Search, X, Bell, Bot, Wrench, Brain, GitBranch, FileText } from 'lucide-react';
import { useNavigate } from 'react-router-dom';

interface SearchResult {
  id: string;
  type: 'notification' | 'skill' | 'agent' | 'memory' | 'workflow';
  title: string;
  subtitle: string;
  path: string;
}

interface GlobalSearchProps {
  onClose: () => void;
}

const typeIcons: Record<string, typeof Search> = {
  notification: Bell,
  skill: Wrench,
  agent: Bot,
  memory: Brain,
  workflow: GitBranch,
};

const typeColors: Record<string, string> = {
  notification: 'text-amber-400 bg-amber-500/10 border-amber-500/20',
  skill: 'text-[#4fc3f7] bg-[#4fc3f7]/10 border-[#4fc3f7]/20',
  agent: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
  memory: 'text-purple-400 bg-purple-500/10 border-purple-500/20',
  workflow: 'text-rose-400 bg-rose-500/10 border-rose-500/20',
};

export default function GlobalSearch({ onClose }: GlobalSearchProps) {
  const [query, setQuery] = useState('');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState<SearchResult[]>([]);
  const [selectedIndex, setSelectedIndex] = useState(0);
  const inputRef = useRef<HTMLInputElement>(null);
  const navigate = useNavigate();

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    if (query.length < 2) {
      setResults([]);
      return;
    }
    const timer = setTimeout(async () => {
      setLoading(true);
      try {
        const all: SearchResult[] = [];
        // Search notifications
        const notifRes = await fetch('/api/notifications');
        if (notifRes.ok) {
          const data = await notifRes.json();
          (data.notifications || []).forEach((n: any) => {
            if (
              n.title?.toLowerCase().includes(query.toLowerCase()) ||
              n.body?.toLowerCase().includes(query.toLowerCase())
            ) {
              all.push({
                id: n.id,
                type: 'notification',
                title: n.title,
                subtitle: n.body?.slice(0, 60) + (n.body?.length > 60 ? '...' : ''),
                path: '/notifications',
              });
            }
          });
        }
        // Search skills
        const skillsRes = await fetch('/api/skills');
        if (skillsRes.ok) {
          const data = await skillsRes.json();
          (data.skills || []).forEach((s: any) => {
            const text = `${s.skill_id || s.id || ''} ${s.description || ''}`.toLowerCase();
            if (text.includes(query.toLowerCase())) {
              all.push({
                id: s.skill_id || s.id,
                type: 'skill',
                title: s.skill_id || s.id,
                subtitle: s.description?.slice(0, 60) || 'Skill',
                path: '/skills',
              });
            }
          });
        }
        // Search workflows
        const wfRes = await fetch('/api/workflows');
        if (wfRes.ok) {
          const data = await wfRes.json();
          (data.workflows || []).forEach((w: any) => {
            const text = `${w.name || w.id || ''}`.toLowerCase();
            if (text.includes(query.toLowerCase())) {
              all.push({
                id: w.id,
                type: 'workflow',
                title: w.name || w.id,
                subtitle: w.status || 'Workflow',
                path: '/workflows',
              });
            }
          });
        }
        // Search agents from status
        const statusRes = await fetch('/api/status');
        if (statusRes.ok) {
          const data = await statusRes.json();
          (data.agents || []).forEach((a: any) => {
            if (a.id?.toLowerCase().includes(query.toLowerCase())) {
              all.push({
                id: a.id,
                type: 'agent',
                title: a.id,
                subtitle: `State: ${a.state}`,
                path: `/agents/${a.id}`,
              });
            }
          });
        }
        setResults(all.slice(0, 20));
        setSelectedIndex(0);
      } catch {
        setResults([]);
      } finally {
        setLoading(false);
      }
    }, 200);
    return () => clearTimeout(timer);
  }, [query]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
        return;
      }
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex((i) => (i + 1) % Math.max(results.length, 1));
        return;
      }
      if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex((i) => (i - 1 + Math.max(results.length, 1)) % Math.max(results.length, 1));
        return;
      }
      if (e.key === 'Enter') {
        e.preventDefault();
        const r = results[selectedIndex];
        if (r) {
          navigate(r.path);
          onClose();
        }
        return;
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [results, selectedIndex, navigate, onClose]);

  return (
    <div className="fixed inset-0 z-[60] flex items-start justify-center pt-[15vh] bg-black/60 backdrop-blur-sm p-4">
      <div className="bg-aurelius-card border border-aurelius-border rounded-xl w-full max-w-xl shadow-2xl overflow-hidden">
        <div className="flex items-center gap-3 px-4 py-3 border-b border-aurelius-border">
          <Search size={18} className="text-aurelius-muted" />
          <input
            ref={inputRef}
            type="text"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search across notifications, skills, agents, workflows..."
            className="flex-1 bg-transparent text-sm text-aurelius-text placeholder:text-aurelius-muted focus:outline-none"
          />
          {loading && (
            <div className="w-4 h-4 border-2 border-aurelius-muted border-t-aurelius-accent rounded-full animate-spin" />
          )}
          <button
            onClick={onClose}
            className="p-1 rounded text-aurelius-muted hover:text-aurelius-text transition-colors"
          >
            <X size={16} />
          </button>
        </div>
        <div className="max-h-[50vh] overflow-y-auto">
          {results.length === 0 && query.length >= 2 && !loading && (
            <div className="px-4 py-8 text-center text-sm text-aurelius-muted">
              <FileText size={24} className="mx-auto mb-2 opacity-40" />
              No results found for "{query}"
            </div>
          )}
          {query.length < 2 && (
            <div className="px-4 py-8 text-center text-sm text-aurelius-muted">
              Type at least 2 characters to search...
            </div>
          )}
          {results.map((r, i) => {
            const Icon = typeIcons[r.type];
            const colorClass = typeColors[r.type];
            return (
              <button
                key={`${r.type}-${r.id}`}
                onClick={() => {
                  navigate(r.path);
                  onClose();
                }}
                className={`w-full flex items-center gap-3 px-4 py-3 text-left transition-colors ${
                  i === selectedIndex ? 'bg-aurelius-accent/10' : 'hover:bg-aurelius-border/20'
                }`}
              >
                <div className={`flex items-center justify-center w-8 h-8 rounded-lg border ${colorClass}`}>
                  <Icon size={14} />
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-aurelius-text truncate">{r.title}</p>
                  <p className="text-xs text-aurelius-muted truncate">{r.subtitle}</p>
                </div>
                <span className="text-[10px] uppercase font-bold text-aurelius-muted bg-aurelius-bg border border-aurelius-border px-1.5 py-0.5 rounded">
                  {r.type}
                </span>
              </button>
            );
          })}
        </div>
        <div className="flex items-center justify-between px-4 py-2 border-t border-aurelius-border text-[10px] text-aurelius-muted">
          <span>{results.length} results</span>
          <span className="flex items-center gap-1">
            <kbd className="px-1 rounded bg-aurelius-bg border border-aurelius-border">↑↓</kbd>
            to navigate
            <kbd className="px-1 rounded bg-aurelius-bg border border-aurelius-border ml-1">Enter</kbd>
            to select
          </span>
        </div>
      </div>
    </div>
  );
}
