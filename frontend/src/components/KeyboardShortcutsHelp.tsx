import { useState, useEffect } from 'react';
import { Keyboard, Command, Search, X } from 'lucide-react';
import Modal from './ui/Modal';

const SHORTCUTS = [
  { keys: ['⌘K', 'Ctrl+K'], desc: 'Command Palette' },
  { keys: ['/'], desc: 'Global Search' },
  { keys: ['G then D'], desc: 'Go to Dashboard' },
  { keys: ['G then C'], desc: 'Go to Chat' },
  { keys: ['G then N'], desc: 'Go to Notifications' },
  { keys: ['G then S'], desc: 'Go to Settings' },
  { keys: ['G then M'], desc: 'Go to Memory' },
  { keys: ['R'], desc: 'Refresh current view' },
  { keys: ['?'], desc: 'Toggle this help' },
];

export default function KeyboardShortcutsHelp() {
  const [open, setOpen] = useState(false);

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === '?' && !e.metaKey && !e.ctrlKey && !e.altKey) {
        const target = e.target as HTMLElement;
        if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) return;
        e.preventDefault();
        setOpen(prev => !prev);
      }
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, []);

  return (
    <>
      <button onClick={() => setOpen(true)} className="aurelius-btn-outline p-2" title="Keyboard Shortcuts">
        <Keyboard size={14} />
      </button>
      {open && (
        <Modal onClose={() => setOpen(false)}>
          <div className="space-y-4">
            <h3 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
              <Keyboard size={18} className="text-[#4fc3f7]" /> Keyboard Shortcuts
            </h3>
            <div className="space-y-2">
              {SHORTCUTS.map(s => (
                <div key={s.desc} className="flex items-center justify-between py-1.5 border-b border-[#2d2d44]/50 last:border-0">
                  <span className="text-sm text-[#e0e0e0]">{s.desc}</span>
                  <div className="flex gap-1.5">
                    {s.keys.map(k => (
                      <kbd key={k} className="text-[10px] font-mono bg-[#0f0f1a] border border-[#2d2d44] px-2 py-0.5 rounded text-[#9e9eb0]">{k}</kbd>
                    ))}
                  </div>
                </div>
              ))}
            </div>
            <p className="text-[10px] text-[#9e9eb0] text-center">Press <kbd className="bg-[#0f0f1a] border border-[#2d2d44] px-1.5 py-0.5 rounded">?</kbd> to toggle this dialog anytime</p>
          </div>
        </Modal>
      )}
    </>
  );
}
