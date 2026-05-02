import { useState, useEffect, useRef, useCallback, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Search,
  LayoutDashboard,
  MessageSquare,
  Bell,
  Wrench,
  GitBranch,
  Brain,
  Settings,
  Zap,
  ArrowRight,
  Loader2,
  X,
} from 'lucide-react';

interface PaletteItem {
  id: string;
  label: string;
  description?: string;
  icon: typeof Search;
  action: () => void;
  shortcut?: string;
  category: string;
}

interface CommandPaletteProps {
  onClose: () => void;
}

const pages = [
  { to: '/', label: 'Dashboard', icon: LayoutDashboard },
  { to: '/chat', label: 'Agent Chat', icon: MessageSquare },
  { to: '/notifications', label: 'Notifications', icon: Bell },
  { to: '/skills', label: 'Skills', icon: Wrench },
  { to: '/workflows', label: 'Workflows', icon: GitBranch },
  { to: '/memory', label: 'Memory', icon: Brain },
  { to: '/settings', label: 'Settings', icon: Settings },
];

const quickCommands = [
  { label: 'Run Health Check', command: 'Run a health check' },
  { label: 'List Active Skills', command: 'List all active skills' },
  { label: 'System Status', command: 'What is the current system status?' },
  { label: 'Show Workflow Status', command: 'Show workflow status' },
];

function fuzzyMatch(query: string, text: string): boolean {
  const q = query.toLowerCase();
  const t = text.toLowerCase();
  let qi = 0;
  for (let ti = 0; ti < t.length && qi < q.length; ti++) {
    if (t[ti] === q[qi]) qi++;
  }
  return qi === q.length;
}

export default function CommandPalette({ onClose }: CommandPaletteProps) {
  const navigate = useNavigate();
  const [query, setQuery] = useState('');
  const [selected, setSelected] = useState(0);
  const [skills, setSkills] = useState<{ id: string; name: string }[]>([]);
  const [workflows, setWorkflows] = useState<{ id: string; name: string }[]>([]);
  const [loading, setLoading] = useState(true);
  const inputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    inputRef.current?.focus();
  }, []);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const [sRes, wRes] = await Promise.all([
          fetch('/api/skills'),
          fetch('/api/workflows'),
        ]);
        if (sRes.ok) {
          const sData = await sRes.json();
          setSkills((sData.skills || []).map((s: { id: string; name: string }) => ({ id: s.id, name: s.name })));
        }
        if (wRes.ok) {
          const wData = await wRes.json();
          setWorkflows((wData.workflows || []).map((w: { id: string; name: string }) => ({ id: w.id, name: w.name })));
        }
      } catch {
        // ignore
      } finally {
        setLoading(false);
      }
    };
    fetchData();
  }, []);

  const items = useMemo<PaletteItem[]>(() => {
    const list: PaletteItem[] = [];

    // Pages
    pages.forEach((p) =>
      list.push({
        id: `page-${p.to}`,
        label: p.label,
        description: 'Navigate to page',
        icon: p.icon,
        action: () => {
          navigate(p.to);
          onClose();
        },
        category: 'Navigation',
      })
    );

    // Quick commands
    quickCommands.forEach((c, i) =>
      list.push({
        id: `cmd-${i}`,
        label: c.label,
        description: c.command,
        icon: Zap,
        action: () => {
          navigate('/chat');
          onClose();
          // Dispatch a custom event for Chat to pick up
          setTimeout(() => {
            window.dispatchEvent(new CustomEvent('aurelius:send-command', { detail: c.command }));
          }, 100);
        },
        category: 'Commands',
      })
    );

    // Skills
    skills.forEach((s) =>
      list.push({
        id: `skill-${s.id}`,
        label: s.name,
        description: `Skill: ${s.id}`,
        icon: Wrench,
        action: () => {
          navigate('/skills');
          onClose();
        },
        category: 'Skills',
      })
    );

    // Workflows
    workflows.forEach((w) =>
      list.push({
        id: `wf-${w.id}`,
        label: w.name,
        description: `Workflow: ${w.id}`,
        icon: GitBranch,
        action: () => {
          navigate('/workflows');
          onClose();
        },
        category: 'Workflows',
      })
    );

    if (!query.trim()) return list;

    return list.filter(
      (item) =>
        fuzzyMatch(query, item.label) ||
        fuzzyMatch(query, item.description || '') ||
        fuzzyMatch(query, item.category)
    );
  }, [query, skills, workflows, navigate, onClose]);

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setSelected(0);
  }, [query]);

  const handleKeyDown = useCallback(
    (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        onClose();
        return;
      }
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelected((prev) => (prev + 1) % items.length);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelected((prev) => (prev - 1 + items.length) % items.length);
      } else if (e.key === 'Enter') {
        e.preventDefault();
        if (items[selected]) {
          items[selected].action();
        }
      }
    },
    [items, selected, onClose]
  );

  useEffect(() => {
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [handleKeyDown]);

  const scrollRef = useRef<HTMLDivElement>(null);
  useEffect(() => {
    const el = scrollRef.current?.children[selected] as HTMLElement | undefined;
    if (el) {
      el.scrollIntoView({ block: 'nearest' });
    }
  }, [selected]);

  return (
    <div
      className="fixed inset-0 z-[60] flex items-start justify-center pt-[15vh] bg-black/50 backdrop-blur-sm"
      onClick={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
    >
      <div className="w-full max-w-xl bg-[#1a1a2e] border border-[#2d2d44] rounded-xl shadow-2xl overflow-hidden">
        {/* Search input */}
        <div className="flex items-center gap-3 px-4 py-3 border-b border-[#2d2d44]">
          <Search size={18} className="text-[#9e9eb0]" />
          <input
            ref={inputRef}
            type="text"
            placeholder="Search pages, skills, workflows, commands..."
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            className="flex-1 bg-transparent text-sm text-[#e0e0e0] placeholder:text-[#9e9eb0] focus:outline-none"
          />
          {loading && <Loader2 size={14} className="animate-spin text-[#9e9eb0]" />}
          <button
            onClick={onClose}
            className="text-[#9e9eb0] hover:text-[#e0e0e0] transition-colors"
          >
            <X size={16} />
          </button>
        </div>

        {/* Results */}
        <div ref={scrollRef} className="max-h-[50vh] overflow-y-auto py-2">
          {items.length === 0 ? (
            <div className="px-4 py-8 text-center text-sm text-[#9e9eb0]">
              No results found for "{query}"
            </div>
          ) : (
            items.map((item, idx) => {
              const Icon = item.icon;
              const isSelected = idx === selected;
              return (
                <button
                  key={item.id}
                  onClick={item.action}
                  onMouseEnter={() => setSelected(idx)}
                  className={`w-full flex items-center gap-3 px-4 py-2.5 text-left transition-colors ${
                    isSelected
                      ? 'bg-[#4fc3f7]/10 text-[#e0e0e0]'
                      : 'text-[#9e9eb0] hover:bg-[#2d2d44]/30'
                  }`}
                >
                  <Icon size={16} className={isSelected ? 'text-[#4fc3f7]' : 'text-[#9e9eb0]'} />
                  <div className="flex-1 min-w-0">
                    <p className="text-sm font-medium truncate">{item.label}</p>
                    {item.description && (
                      <p className="text-xs opacity-70 truncate">{item.description}</p>
                    )}
                  </div>
                  <span className="text-[10px] uppercase tracking-wider px-1.5 py-0.5 rounded bg-[#2d2d44]/30 border border-[#2d2d44]/40 shrink-0">
                    {item.category}
                  </span>
                  {isSelected && <ArrowRight size={14} className="text-[#4fc3f7]" />}
                </button>
              );
            })
          )}
        </div>

        {/* Footer hints */}
        <div className="flex items-center gap-4 px-4 py-2 border-t border-[#2d2d44] text-[10px] text-[#9e9eb0]">
          <span className="flex items-center gap-1">
            <kbd className="px-1 py-0.5 rounded bg-[#2d2d44]/40 border border-[#2d2d44]/60 font-mono">↑↓</kbd>
            Navigate
          </span>
          <span className="flex items-center gap-1">
            <kbd className="px-1 py-0.5 rounded bg-[#2d2d44]/40 border border-[#2d2d44]/60 font-mono">Enter</kbd>
            Select
          </span>
          <span className="flex items-center gap-1">
            <kbd className="px-1 py-0.5 rounded bg-[#2d2d44]/40 border border-[#2d2d44]/60 font-mono">Esc</kbd>
            Close
          </span>
        </div>
      </div>
    </div>
  );
}
