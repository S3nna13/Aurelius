import { useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import {
  LayoutDashboard, MessageSquare, Bell, Wrench, GitBranch, Brain, Settings,
  Menu, X, Shield, ScrollText, CalendarClock, BarChart3, HeartPulse,
  BookOpen, LineChart, Cpu, Bot, Database, History, Puzzle, Sparkles,
  Play, Layers, Globe, Terminal, ChevronDown, ChevronRight,
} from 'lucide-react';
import { useApi } from '../hooks/useApi';

const mainNav = [
  { to: '/', icon: LayoutDashboard, label: 'Dashboard' },
  { to: '/chat', icon: MessageSquare, label: 'Agent Chat' },
  { to: '/analytics', icon: BarChart3, label: 'Analytics' },
  { to: '/notifications', icon: Bell, label: 'Notifications', badgeKey: 'notifications' as const },
];

const agentNav = [
  { to: '/agents/grid', icon: Bot, label: 'Agent Grid' },
  { to: '/agents/playground', icon: Play, label: 'Playground' },
  { to: '/agents/skills', icon: Puzzle, label: 'Skill Builder' },
  { to: '/agents/templates', icon: Sparkles, label: 'Templates' },
  { to: '/agents/knowledge', icon: Globe, label: 'Knowledge Base' },
  { to: '/agents/tools', icon: Terminal, label: 'Tool Registry' },
];

const systemNav = [
  { to: '/traces', icon: History, label: 'Traces' },
  { to: '/training', icon: LineChart, label: 'Training' },
  { to: '/workflows', icon: GitBranch, label: 'Workflows' },
  { to: '/memory', icon: Brain, label: 'Memory' },
  { to: '/tasks', icon: CalendarClock, label: 'Tasks' },
  { to: '/skills', icon: Wrench, label: 'Skills' },
  { to: '/plugins', icon: Layers, label: 'Plugins' },
  { to: '/models', icon: Cpu, label: 'Models' },
  { to: '/data', icon: Database, label: 'Data Explorer' },
];

const adminNav = [
  { to: '/users', icon: Shield, label: 'Users' },
  { to: '/logs', icon: ScrollText, label: 'Logs' },
  { to: '/playground', icon: Bot, label: 'Playground' },
  { to: '/health', icon: HeartPulse, label: 'Health' },
  { to: '/api-docs', icon: BookOpen, label: 'API Docs' },
  { to: '/settings', icon: Settings, label: 'Settings' },
];

export default function Sidebar() {
  const [open, setOpen] = useState(false);
  const [agentsOpen, setAgentsOpen] = useState(true);
  const location = useLocation();

  const { data: notifStats } = useApi<{ unread: number }>('/notifications/stats', { refreshInterval: 15000 });
  const { data: workflowData } = useApi<{ workflows: { status: string }[] }>('/workflows', { refreshInterval: 15000 });

  const runningCount = workflowData?.workflows?.filter(w => w.status === 'running').length ?? 0;
  const unreadCount = notifStats?.unread ?? 0;
  const badges: Record<string, number> = { notifications: unreadCount, workflows: runningCount };

  const isActive = (path: string) => {
    if (path === '/') return location.pathname === '/';
    return location.pathname.startsWith(path);
  };

  const NavItem = ({ to, icon: Icon, label, badgeKey }: { to: string; icon: any; label: string; badgeKey?: string }) => {
    const active = isActive(to);
    const badgeCount = badgeKey ? (badges[badgeKey] || 0) : 0;
    return (
      <NavLink to={to} onClick={() => setOpen(false)}
        className={`flex items-center gap-3 px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
          active ? 'bg-[#4fc3f7]/10 text-[#4fc3f7] border border-[#4fc3f7]/20' : 'text-[#9e9eb0] hover:text-[#e0e0e0] hover:bg-white/5'
        }`}
      >
        <Icon size={16} />
        <span className="flex-1">{label}</span>
        {badgeCount > 0 && (
          <span className={`flex h-5 min-w-[20px] items-center justify-center rounded-full px-1.5 text-[10px] font-bold ${badgeKey === 'notifications' ? 'bg-red-500 text-white' : 'bg-[#4fc3f7] text-[#05050f]'}`}>
            {badgeCount > 99 ? '99+' : badgeCount}
          </span>
        )}
      </NavLink>
    );
  };

  return (
    <>
      <button className="fixed top-4 left-4 z-50 md:hidden text-[#4fc3f7] bg-[#0f0f1a] border border-[#2d2d44] rounded-lg p-2" onClick={() => setOpen(!open)} aria-label="Toggle sidebar">
        {open ? <X size={20} /> : <Menu size={20} />}
      </button>

      {open && <div className="fixed inset-0 bg-black/60 z-30 md:hidden" onClick={() => setOpen(false)} />}

      <aside className={`fixed top-0 left-0 z-40 h-full w-64 bg-[#0f0f1a] border-r border-[#2d2d44] transform transition-transform duration-300 ease-in-out ${open ? 'translate-x-0' : '-translate-x-full'} md:translate-x-0 flex flex-col`}>
        <div className="flex items-center gap-3 px-6 py-5 border-b border-[#2d2d44]">
          <Shield className="text-[#4fc3f7]" size={24} />
          <div>
            <h1 className="text-lg font-bold text-[#e0e0e0] tracking-wide">AURELIUS</h1>
            <p className="text-xs text-[#9e9eb0] uppercase tracking-wider">Operations Center</p>
          </div>
        </div>

        <nav className="flex-1 px-3 py-4 space-y-1 overflow-y-auto">
          <p className="text-[10px] font-bold text-[#9e9eb0] uppercase tracking-wider px-3 py-1.5">Main</p>
          {mainNav.map(item => <NavItem key={item.to} {...item} />)}

          <div className="pt-3">
            <button onClick={() => setAgentsOpen(!agentsOpen)} className="flex items-center gap-2 px-3 py-1.5 text-[10px] font-bold text-[#9e9eb0] uppercase tracking-wider w-full hover:text-[#e0e0e0]">
              {agentsOpen ? <ChevronDown size={12} /> : <ChevronRight size={12} />}
              Agents ({agentNav.length})
            </button>
            {agentsOpen && <div className="space-y-0.5 pl-3 pt-1">{agentNav.map(item => <NavItem key={item.to} {...item} />)}</div>}
          </div>

          <div className="pt-3">
            <p className="text-[10px] font-bold text-[#9e9eb0] uppercase tracking-wider px-3 py-1.5">System</p>
            {systemNav.map(item => <NavItem key={item.to} {...item} />)}
          </div>

          <div className="pt-3">
            <p className="text-[10px] font-bold text-[#9e9eb0] uppercase tracking-wider px-3 py-1.5">Administration</p>
            {adminNav.map(item => <NavItem key={item.to} {...item} />)}
          </div>
        </nav>

        <div className="px-6 py-4 border-t border-[#2d2d44] text-xs text-[#9e9eb0]">
          <p>Aurelius Operations Center</p>
          <p>v2.0.0 · {agentNav.length + systemNav.length + mainNav.length + adminNav.length} modules</p>
        </div>
      </aside>
    </>
  );
}
