import { useEffect, useState } from 'react';
import { X, Command, Keyboard } from 'lucide-react';

interface Shortcut {
  keys: string[];
  description: string;
  category: string;
}

const shortcuts: Shortcut[] = [
  { keys: ['?'], description: 'Show keyboard shortcuts', category: 'General' },
  { keys: ['Cmd', 'K'], description: 'Open command palette', category: 'General' },
  { keys: ['Esc'], description: 'Close modal or palette', category: 'General' },
  { keys: ['/'], description: 'Focus search (when available)', category: 'General' },
  { keys: ['G', 'D'], description: 'Go to Dashboard', category: 'Navigation' },
  { keys: ['G', 'C'], description: 'Go to Chat', category: 'Navigation' },
  { keys: ['G', 'N'], description: 'Go to Notifications', category: 'Navigation' },
  { keys: ['G', 'S'], description: 'Go to Settings', category: 'Navigation' },
  { keys: ['R'], description: 'Refresh current page data', category: 'Actions' },
  { keys: ['T'], description: 'Toggle theme', category: 'Actions' },
];

export default function KeyboardShortcuts() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === '?' && !e.metaKey && !e.ctrlKey && !e.altKey) {
        e.preventDefault();
        setOpen((prev) => !prev);
      }
      if (e.key === 'Escape') {
        setOpen(false);
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  if (!open) return null;

  const categories = [...new Set(shortcuts.map((s) => s.category))];

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="bg-aurelius-card border border-aurelius-border rounded-xl w-full max-w-lg shadow-2xl">
        <div className="flex items-center justify-between px-6 py-4 border-b border-aurelius-border">
          <div className="flex items-center gap-2">
            <Keyboard size={18} className="text-aurelius-accent" />
            <h3 className="text-base font-bold text-aurelius-text">Keyboard Shortcuts</h3>
          </div>
          <button
            onClick={() => setOpen(false)}
            className="p-1.5 rounded-lg text-aurelius-muted hover:text-aurelius-text hover:bg-aurelius-border/40 transition-colors"
          >
            <X size={16} />
          </button>
        </div>
        <div className="px-6 py-4 space-y-5 max-h-[60vh] overflow-y-auto">
          {categories.map((cat) => (
            <div key={cat}>
              <h4 className="text-xs font-semibold text-aurelius-muted uppercase tracking-wider mb-2">
                {cat}
              </h4>
              <div className="space-y-2">
                {shortcuts
                  .filter((s) => s.category === cat)
                  .map((s, i) => (
                    <div key={i} className="flex items-center justify-between">
                      <span className="text-sm text-aurelius-text">{s.description}</span>
                      <div className="flex items-center gap-1">
                        {s.keys.map((k, ki) => (
                          <span key={ki} className="flex items-center">
                            <kbd className="px-1.5 py-0.5 text-[10px] font-bold bg-aurelius-bg border border-aurelius-border rounded text-aurelius-muted min-w-[20px] text-center">
                              {k === 'Cmd' ? <Command size={10} className="inline" /> : k}
                            </kbd>
                            {ki < s.keys.length - 1 && (
                              <span className="text-aurelius-muted mx-0.5 text-[10px]">+</span>
                            )}
                          </span>
                        ))}
                      </div>
                    </div>
                  ))}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
