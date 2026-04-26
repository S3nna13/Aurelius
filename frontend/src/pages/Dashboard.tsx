import {
  Bot,
  CheckCircle2,
  Wrench,
  HeartPulse,
  Activity,
  Zap,
  RotateCcw,
  Pause,
  Play,
  Terminal,
} from 'lucide-react';

const statusCards = [
  {
    label: 'Agents Online',
    value: '4',
    icon: Bot,
    color: 'text-emerald-400',
    bg: 'bg-emerald-500/10',
    border: 'border-emerald-500/20',
  },
  {
    label: 'Tasks Completed Today',
    value: '127',
    icon: CheckCircle2,
    color: 'text-aurelius-accent',
    bg: 'bg-aurelius-accent/10',
    border: 'border-aurelius-accent/20',
  },
  {
    label: 'Skills Active',
    value: '18',
    icon: Wrench,
    color: 'text-amber-400',
    bg: 'bg-amber-500/10',
    border: 'border-amber-500/20',
  },
  {
    label: 'System Health',
    value: '98%',
    icon: HeartPulse,
    color: 'text-rose-400',
    bg: 'bg-rose-500/10',
    border: 'border-rose-500/20',
  },
];

const agents = [
  { name: 'Hermes', role: 'Notification Router', status: 'active' as const },
  { name: 'OpenClaw', role: 'Task Orchestrator', status: 'active' as const },
  { name: 'Cerebrum', role: 'Memory Manager', status: 'idle' as const },
  { name: 'Vigil', role: 'Security Warden', status: 'active' as const },
];

const activities = [
  { time: '2 min ago', message: 'OpenClaw completed workflow "Daily Backup"', type: 'success' as const },
  { time: '5 min ago', message: 'Hermes dispatched alert: High CPU usage on node-2', type: 'warning' as const },
  { time: '12 min ago', message: 'Cerebrum pruned 42 stale memory entries', type: 'info' as const },
  { time: '18 min ago', message: 'Vigil blocked unauthorized access attempt', type: 'error' as const },
  { time: '25 min ago', message: 'New skill "Web Scraping" registered', type: 'success' as const },
  { time: '31 min ago', message: 'Agent OpenClaw restarted after update', type: 'info' as const },
  { time: '45 min ago', message: 'Memory layer "short-term" compacted', type: 'info' as const },
  { time: '1 hr ago', message: 'Workflow "Data Ingest" failed — retrying', type: 'warning' as const },
];

const statusBadge = (status: string) => {
  switch (status) {
    case 'active':
      return (
        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 animate-pulse" />
          Active
        </span>
      );
    case 'idle':
      return (
        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-amber-500/10 text-amber-400 border border-amber-500/20">
          <span className="w-1.5 h-1.5 rounded-full bg-amber-400" />
          Idle
        </span>
      );
    default:
      return (
        <span className="inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium bg-red-500/10 text-red-400 border border-red-500/20">
          <span className="w-1.5 h-1.5 rounded-full bg-red-400" />
          Error
        </span>
      );
  }
};

const activityIcon = (type: string) => {
  switch (type) {
    case 'success':
      return <CheckCircle2 size={14} className="text-emerald-400 shrink-0 mt-0.5" />;
    case 'warning':
      return <Zap size={14} className="text-amber-400 shrink-0 mt-0.5" />;
    case 'error':
      return <HeartPulse size={14} className="text-red-400 shrink-0 mt-0.5" />;
    default:
      return <Activity size={14} className="text-aurelius-accent shrink-0 mt-0.5" />;
  }
};

export default function Dashboard() {
  return (
    <div className="space-y-6">
      {/* Status Cards */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-4">
        {statusCards.map((card) => (
          <div
            key={card.label}
            className="aurelius-card flex items-center gap-4 hover:border-aurelius-accent/30 transition-colors"
          >
            <div
              className={`flex items-center justify-center w-10 h-10 rounded-lg ${card.bg} ${card.color} border ${card.border}`}
            >
              <card.icon size={20} />
            </div>
            <div>
              <p className="text-2xl font-bold text-aurelius-text">{card.value}</p>
              <p className="text-xs text-aurelius-muted">{card.label}</p>
            </div>
          </div>
        ))}
      </div>

      {/* Middle Row */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Agent Status List */}
        <div className="lg:col-span-2 aurelius-card space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-aurelius-text uppercase tracking-wider flex items-center gap-2">
              <Bot size={16} className="text-aurelius-accent" />
              Agent Status
            </h3>
            <span className="text-xs text-aurelius-muted">4 agents registered</span>
          </div>

          <div className="space-y-2">
            {agents.map((agent) => (
              <div
                key={agent.name}
                className="flex items-center justify-between px-3 py-2.5 rounded-lg bg-aurelius-bg/50 border border-aurelius-border/50 hover:border-aurelius-accent/20 transition-colors"
              >
                <div className="flex items-center gap-3">
                  <div className="w-8 h-8 rounded-full bg-aurelius-border/40 flex items-center justify-center text-aurelius-accent">
                    <Bot size={16} />
                  </div>
                  <div>
                    <p className="text-sm font-medium text-aurelius-text">{agent.name}</p>
                    <p className="text-xs text-aurelius-muted">{agent.role}</p>
                  </div>
                </div>
                {statusBadge(agent.status)}
              </div>
            ))}
          </div>
        </div>

        {/* Quick Actions */}
        <div className="aurelius-card space-y-4">
          <h3 className="text-sm font-semibold text-aurelius-text uppercase tracking-wider flex items-center gap-2">
            <Terminal size={16} className="text-aurelius-accent" />
            Quick Actions
          </h3>
          <div className="grid grid-cols-2 gap-3">
            <button className="aurelius-btn flex items-center justify-center gap-2 text-sm">
              <Play size={14} />
              Start All
            </button>
            <button className="aurelius-btn-outline flex items-center justify-center gap-2 text-sm">
              <Pause size={14} />
              Pause All
            </button>
            <button className="aurelius-btn-outline flex items-center justify-center gap-2 text-sm">
              <RotateCcw size={14} />
              Restart
            </button>
            <button className="aurelius-btn-outline flex items-center justify-center gap-2 text-sm">
              <Zap size={14} />
              Run Task
            </button>
          </div>

          <div className="pt-2 border-t border-aurelius-border">
            <p className="text-xs text-aurelius-muted mb-2">System Uptime</p>
            <p className="text-lg font-mono font-bold text-aurelius-accent">3d 14h 22m</p>
          </div>
        </div>
      </div>

      {/* Recent Activity Feed */}
      <div className="aurelius-card space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-aurelius-text uppercase tracking-wider flex items-center gap-2">
            <Activity size={16} className="text-aurelius-accent" />
            Recent Activity
          </h3>
          <button className="text-xs text-aurelius-accent hover:underline">View all</button>
        </div>

        <div className="space-y-2 max-h-80 overflow-y-auto pr-1">
          {activities.map((act, idx) => (
            <div
              key={idx}
              className="flex items-start gap-3 px-3 py-2 rounded-lg bg-aurelius-bg/50 border border-aurelius-border/50 hover:border-aurelius-accent/10 transition-colors"
            >
              {activityIcon(act.type)}
              <div className="flex-1 min-w-0">
                <p className="text-sm text-aurelius-text leading-snug">{act.message}</p>
                <p className="text-xs text-aurelius-muted mt-0.5">{act.time}</p>
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
