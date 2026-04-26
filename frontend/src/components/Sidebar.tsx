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
  ScrollText,
  CalendarClock,
  BarChart3,
  HeartPulse,
  BookOpen,
  LineChart,
  Cpu,
} from 'lucide-react';
import { useApi } from '../hooks/useApi';

const navItems = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard', badge: null as string | null },
  { to: '/chat', icon: MessageSquare, label: 'Agent Chat', badge: null },
  { to: '/notifications', icon: Bell, label: 'Notifications', badgeKey: 'notifications' as const },
  { to: '/skills', icon: Wrench, label: 'Skills', badge: null },
  { to: '/workflows', icon: GitBranch, label: 'Workflows', badgeKey: 'workflows' as const },
  { to: '/memory', icon: Brain, label: 'Memory', badge: null },
  { to: '/tasks', icon: CalendarClock, label: 'Tasks', badge: null },
  { to: '/training', icon: LineChart, label: 'Training', badge: null },
  { to: '/models', icon: Cpu, label: 'Models', badge: null },
  { to: '/agents', icon: BarChart3, label: 'Agents', badge: null },
  { to: '/settings', icon: Settings, label: 'Settings', badge: null },
  { to: '/health', icon: HeartPulse, label: 'Health', badge: null },
  { to: '/api-docs', icon: BookOpen, label: 'API Docs', badge: null },
  { to: '/logs', icon: ScrollText, label: 'Logs', badge: null },
];

export default function Sidebar() {
  const [open, setOpen] = useState(false);
  const location = useLocation();

  const { data: notifStats } = useApi<{ unread: number }>('/notifications/stats', {
    refreshInterval: 15000,
  });

  const { data: workflowData } = useApi<{ workflows: { status: string }[] }>('/workflows', {
    refreshInterval: 15000,
  });

  const runningCount = workflowData?.workflows?.filter((w) => w.status === 'running').length ?? 0;
  const unreadCount = notifStats?.unread ?? 0;

  const badges: Record<string, number> = {
    notifications: unreadCount,
    workflows: runningCount,
  };

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
            const badgeCount = item.badgeKey ? badges[item.badgeKey] || 0 : 0;
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
                <span className="flex-1">{item.label}</span>
                {badgeCount > 0 && (
                  <span className={`
                    flex h-5 min-w-[20px] items-center justify-center rounded-full px-1.5 text-[10px] font-bold
                    ${item.badgeKey === 'notifications'
                      ? 'bg-red-500 text-white'
                      : 'bg-aurelius-accent text-aurelius-bg'
                    }
                  `}>
                    {badgeCount > 99 ? '99+' : badgeCount}
                  </span>
                )}
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
