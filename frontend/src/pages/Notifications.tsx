import { useState } from 'react';
import {
  Bell,
  CheckCheck,
  AlertTriangle,
  Info,
  AlertCircle,
  Bot,
  Server,
  ShieldAlert,
} from 'lucide-react';

type Channel = 'all' | 'agent' | 'system' | 'alerts';
type Priority = 'high' | 'medium' | 'low';

interface Notification {
  id: number;
  title: string;
  message: string;
  channel: Exclude<Channel, 'all'>;
  priority: Priority;
  read: boolean;
  time: string;
}

const initialNotifications: Notification[] = [
  {
    id: 1,
    title: 'High CPU Usage',
    message: 'Node-2 CPU usage exceeded 85% for more than 5 minutes.',
    channel: 'alerts',
    priority: 'high',
    read: false,
    time: '2 min ago',
  },
  {
    id: 2,
    title: 'Agent OpenClaw Completed Task',
    message: 'Workflow "Daily Backup" finished successfully in 42s.',
    channel: 'agent',
    priority: 'low',
    read: false,
    time: '5 min ago',
  },
  {
    id: 3,
    title: 'Memory Prune Complete',
    message: 'Cerebrum pruned 42 stale entries from short-term memory.',
    channel: 'system',
    priority: 'low',
    read: true,
    time: '12 min ago',
  },
  {
    id: 4,
    title: 'Unauthorized Access Blocked',
    message: 'Vigil blocked an unauthorized login attempt from 192.168.1.45.',
    channel: 'alerts',
    priority: 'high',
    read: false,
    time: '18 min ago',
  },
  {
    id: 5,
    title: 'New Skill Registered',
    message: '"Web Scraping" skill v2.1.0 was enabled by the admin.',
    channel: 'system',
    priority: 'medium',
    read: true,
    time: '25 min ago',
  },
  {
    id: 6,
    title: 'Agent Restarted',
    message: 'OpenClaw was restarted after a config update.',
    channel: 'agent',
    priority: 'medium',
    read: false,
    time: '31 min ago',
  },
  {
    id: 7,
    title: 'Workflow Failure',
    message: '"Data Ingest" workflow failed on step 3. Auto-retry in progress.',
    channel: 'alerts',
    priority: 'high',
    read: false,
    time: '1 hr ago',
  },
  {
    id: 8,
    title: 'Disk Space Warning',
    message: 'Node-1 disk usage at 78%. Consider cleanup.',
    channel: 'system',
    priority: 'medium',
    read: true,
    time: '2 hr ago',
  },
];

const tabs: { key: Channel; label: string }[] = [
  { key: 'all', label: 'All' },
  { key: 'agent', label: 'Agent' },
  { key: 'system', label: 'System' },
  { key: 'alerts', label: 'Alerts' },
];

const priorityConfig: Record<
  Priority,
  { color: string; bg: string; border: string; icon: typeof AlertTriangle }
> = {
  high: {
    color: 'text-red-400',
    bg: 'bg-red-500/10',
    border: 'border-red-500/20',
    icon: ShieldAlert,
  },
  medium: {
    color: 'text-amber-400',
    bg: 'bg-amber-500/10',
    border: 'border-amber-500/20',
    icon: AlertTriangle,
  },
  low: {
    color: 'text-aurelius-accent',
    bg: 'bg-aurelius-accent/10',
    border: 'border-aurelius-accent/20',
    icon: Info,
  },
};

const channelIcon = (channel: string) => {
  switch (channel) {
    case 'agent':
      return <Bot size={14} />;
    case 'system':
      return <Server size={14} />;
    case 'alerts':
      return <AlertCircle size={14} />;
    default:
      return <Bell size={14} />;
  }
};

export default function Notifications() {
  const [activeTab, setActiveTab] = useState<Channel>('all');
  const [notifications, setNotifications] = useState<Notification[]>(initialNotifications);

  const filtered =
    activeTab === 'all'
      ? notifications
      : notifications.filter((n) => n.channel === activeTab);

  const unreadCount = notifications.filter((n) => !n.read).length;

  const markAllRead = () => {
    setNotifications((prev) => prev.map((n) => ({ ...n, read: true })));
  };

  const toggleRead = (id: number) => {
    setNotifications((prev) =>
      prev.map((n) => (n.id === id ? { ...n, read: !n.read } : n))
    );
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-aurelius-text flex items-center gap-2">
            <Bell size={20} className="text-aurelius-accent" />
            Hermes Notification Center
          </h2>
          <p className="text-sm text-aurelius-muted mt-0.5">
            {unreadCount} unread notification{unreadCount !== 1 ? 's' : ''}
          </p>
        </div>
        <button
          onClick={markAllRead}
          className="aurelius-btn-outline flex items-center gap-2 text-sm self-start sm:self-auto"
        >
          <CheckCheck size={14} />
          Mark all as read
        </button>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-aurelius-border pb-1 overflow-x-auto">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors whitespace-nowrap ${
              activeTab === tab.key
                ? 'text-aurelius-accent border-b-2 border-aurelius-accent bg-aurelius-accent/5'
                : 'text-aurelius-muted hover:text-aurelius-text'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* List */}
      <div className="space-y-2">
        {filtered.length === 0 && (
          <div className="aurelius-card text-center py-12 text-aurelius-muted">
            <Bell size={32} className="mx-auto mb-3 opacity-40" />
            <p>No notifications in this channel.</p>
          </div>
        )}
        {filtered.map((n) => {
          const p = priorityConfig[n.priority];
          return (
            <div
              key={n.id}
              onClick={() => toggleRead(n.id)}
              className={`
                aurelius-card flex items-start gap-4 cursor-pointer transition-all
                ${n.read ? 'opacity-70' : 'opacity-100'}
                hover:border-aurelius-accent/30
              `}
            >
              {/* Unread indicator */}
              <div className="pt-1.5">
                <div
                  className={`w-2.5 h-2.5 rounded-full ${
                    n.read ? 'bg-aurelius-border' : 'bg-aurelius-accent animate-pulse'
                  }`}
                />
              </div>

              {/* Priority icon */}
              <div
                className={`flex items-center justify-center w-9 h-9 rounded-lg shrink-0 ${p.bg} ${p.color} border ${p.border}`}
              >
                <p.icon size={16} />
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 mb-0.5">
                  <h3
                    className={`text-sm ${
                      n.read ? 'font-medium text-aurelius-muted' : 'font-semibold text-aurelius-text'
                    }`}
                  >
                    {n.title}
                  </h3>
                  <span
                    className={`text-[10px] uppercase font-bold px-1.5 py-0.5 rounded ${p.bg} ${p.color} border ${p.border}`}
                  >
                    {n.priority}
                  </span>
                  <span className="flex items-center gap-1 text-[10px] text-aurelius-muted bg-aurelius-bg px-1.5 py-0.5 rounded border border-aurelius-border">
                    {channelIcon(n.channel)}
                    {n.channel}
                  </span>
                </div>
                <p className="text-sm text-aurelius-muted leading-relaxed">{n.message}</p>
                <p className="text-xs text-aurelius-muted/60 mt-1">{n.time}</p>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
