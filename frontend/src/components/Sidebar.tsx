import { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  LayoutDashboard,
  MessageSquare,
  Bell,
  Wrench,
  GitBranch,
  Brain,
  Settings,
  Menu,
  X,
  Shield,
} from 'lucide-react';

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/chat', icon: MessageSquare, label: 'Agent Chat' },
  { to: '/notifications', icon: Bell, label: 'Notifications' },
  { to: '/skills', icon: Wrench, label: 'Skills' },
  { to: '/workflows', icon: GitBranch, label: 'Workflows' },
  { to: '/memory', icon: Brain, label: 'Memory' },
  { to: '/settings', icon: Settings, label: 'Settings' },
];

export default function Sidebar() {
  const [open, setOpen] = useState(false);
  const location = useLocation();

  const isActive = (path: string) => location.pathname === path;

  return (
    <>
      {/* Mobile hamburger */}
      <button
        className="fixed top-4 left-4 z-50 md:hidden text-aurelius-accent bg-aurelius-card border border-aurelius-border rounded-lg p-2"
        onClick={() => setOpen(!open)}
        aria-label="Toggle sidebar"
      >
        {open ? <X size={20} /> : <Menu size={20} />}
      </button>

      {/* Overlay */}
      {open && (
        <div
          className="fixed inset-0 bg-black/60 z-30 md:hidden"
          onClick={() => setOpen(false)}
        />
      )}

      {/* Sidebar */}
      <aside
        className={`
          fixed top-0 left-0 z-40 h-full w-64 bg-aurelius-card border-r border-aurelius-border
          transform transition-transform duration-300 ease-in-out
          ${open ? 'translate-x-0' : '-translate-x-full'} md:translate-x-0
          flex flex-col
        `}
      >
        {/* Logo */}
        <div className="flex items-center gap-3 px-6 py-5 border-b border-aurelius-border">
          <Shield className="text-aurelius-accent" size={24} />
          <div>
            <h1 className="text-lg font-bold text-aurelius-text tracking-wide">AURELIUS</h1>
            <p className="text-xs text-aurelius-muted uppercase tracking-wider">Mission Control</p>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          {navItems.map((item) => {
            const active = isActive(item.to);
            return (
              <NavLink
                key={item.to}
                to={item.to}
                onClick={() => setOpen(false)}
                className={`
                  flex items-center gap-3 px-3 py-2.5 rounded-lg text-sm font-medium transition-colors
                  ${active
                    ? 'bg-aurelius-accent/10 text-aurelius-accent border border-aurelius-accent/20'
                    : 'text-aurelius-muted hover:text-aurelius-text hover:bg-aurelius-border/40'
                  }
                `}
              >
                <item.icon size={18} />
                <span>{item.label}</span>
              </NavLink>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="px-6 py-4 border-t border-aurelius-border text-xs text-aurelius-muted">
          <p>Hermes-OpenClaw</p>
          <p>v1.0.0</p>
        </div>
      </aside>
    </>
  );
}
