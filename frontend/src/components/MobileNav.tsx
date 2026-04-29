import { useState } from 'react';
import { Menu, X, Bot } from 'lucide-react';
import { useLocation } from 'react-router-dom';

const NAV_ITEMS = [
  { path: '/', label: 'Dashboard' }, { path: '/agents/grid', label: 'Agents' },
  { path: '/chat', label: 'Chat' }, { path: '/traces', label: 'Traces' },
  { path: '/training', label: 'Training' }, { path: '/analytics', label: 'Analytics' },
  { path: '/skills', label: 'Skills' }, { path: '/settings', label: 'Settings' },
];

export default function MobileNav() {
  const [open, setOpen] = useState(false);
  const location = useLocation();

  return (
    <div className="md:hidden">
      <button onClick={() => setOpen(true)} className="p-2 text-[#9e9eb0]">
        <Menu size={20} />
      </button>
      {open && (
        <div className="fixed inset-0 z-50 bg-[#05050f]/95 backdrop-blur-md">
          <div className="p-4">
            <div className="flex items-center justify-between mb-8">
              <div className="flex items-center gap-2">
                <Bot size={20} className="text-[#4fc3f7]" />
                <span className="font-bold text-[#e0e0e0]">Aurelius</span>
              </div>
              <button onClick={() => setOpen(false)} className="text-[#9e9eb0]"><X size={20} /></button>
            </div>
            <div className="space-y-1">
              {NAV_ITEMS.map(item => {
                const active = location.pathname === item.path || (item.path !== '/' && location.pathname.startsWith(item.path));
                return (
                  <a key={item.path} href={item.path} onClick={() => setOpen(false)}
                    className={`block px-4 py-3 rounded-lg text-sm transition-colors ${active ? 'bg-[#4fc3f7]/10 text-[#4fc3f7]' : 'text-[#9e9eb0] hover:text-[#e0e0e0]'}`}
                  >
                    {item.label}
                  </a>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
