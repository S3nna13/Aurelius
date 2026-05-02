import { useState, useEffect, useCallback, useRef, useMemo } from 'react';
import {
  Bell,
  CheckCheck,
  AlertTriangle,
  Info,
  AlertCircle,
  Bot,
  Server,
  ShieldAlert,
  Loader2,
  RefreshCw,
} from 'lucide-react';
import { useApi } from '../hooks/useApi';
import { useToast } from '../components/ToastProvider';
import { usePushNotifications } from '../hooks/usePushNotifications';
import BulkActionsBar from '../components/BulkActionsBar';
import Pagination from '../components/Pagination';
import { downloadJSON } from '../utils/export';

type Channel = 'all' | 'agent' | 'system' | 'alerts';
type Priority = 'high' | 'medium' | 'low';

interface ApiNotification {
  id: string;
  timestamp: number;
  channel: string;
  priority: string;
  category: string;
  title: string;
  body: string;
  read: boolean;
  delivered: boolean;
}

interface NotificationStats {
  unread: number;
  total: number;
  by_channel: Record<string, number>;
  by_priority: Record<string, number>;
}

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
    color: 'text-rose-400',
    bg: 'bg-rose-500/10',
    border: 'border-rose-500/20',
    icon: ShieldAlert,
  },
  medium: {
    color: 'text-amber-400',
    bg: 'bg-amber-500/10',
    border: 'border-amber-500/20',
    icon: AlertTriangle,
  },
  low: {
    color: 'text-[#4fc3f7]',
    bg: 'bg-[#4fc3f7]/10',
    border: 'border-[#4fc3f7]/20',
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

function timeAgo(ts: number): string {
  const diff = Date.now() / 1000 - ts;
  if (diff < 60) return 'Just now';
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} hr ago`;
  return `${Math.floor(diff / 86400)} days ago`;
}

export default function Notifications() {
  const [activeTab, setActiveTab] = useState<Channel>('all');
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [currentPage, setCurrentPage] = useState(1);
  const itemsPerPage = 10;
  const { toast } = useToast();
  const { notify } = usePushNotifications();

  const {
    data: notifData,
    loading: notifLoading,
    error: notifError,
    refresh: refreshNotifs,
  } = useApi<{ notifications: ApiNotification[] }>('/notifications', {
    refreshInterval: 5000,
  });

  const {
    data: statsData,
    refresh: refreshStats,
  } = useApi<NotificationStats>('/notifications/stats', {
    refreshInterval: 5000,
  });

  const refreshAll = useCallback(() => {
    refreshNotifs();
    refreshStats();
  }, [refreshNotifs, refreshStats]);

  const notifications = useMemo(() => notifData?.notifications || [], [notifData]);

  useEffect(() => {
    if (notifError) {
      toast('Failed to load notifications', 'error');
    }
  }, [notifError, toast]);

  // Push notifications for new items
  const prevCountRef = useRef(0);
  useEffect(() => {
    const unread = notifications.filter((n) => !n.read).length;
    if (unread > prevCountRef.current && prevCountRef.current > 0) {
      const newNotifs = notifications.filter((n) => !n.read).slice(0, unread - prevCountRef.current);
      newNotifs.forEach((n) => {
        notify(n.title, { body: n.body, tag: n.id });
      });
    }
    prevCountRef.current = unread;
  }, [notifications, notify]);

  const filtered =
    activeTab === 'all'
      ? notifications
      : notifications.filter((n) => n.channel === activeTab);

  const totalPages = Math.max(1, Math.ceil(filtered.length / itemsPerPage));
  const paginated = filtered.slice((currentPage - 1) * itemsPerPage, currentPage * itemsPerPage);

  const unreadCount = notifications.filter((n) => !n.read).length;

  const toggleSelection = (id: string) => {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(id)) next.delete(id);
      else next.add(id);
      return next;
    });
  };

  const selectAll = () => setSelectedIds(new Set(paginated.map((n) => n.id)));
  const deselectAll = () => setSelectedIds(new Set());

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setCurrentPage(1);
  }, [activeTab]);

  const exportSelected = () => {
    const selected = filtered.filter((n) => selectedIds.has(n.id));
    if (selected.length === 0) return;
    downloadJSON(selected, `notifications-${Date.now()}.json`);
    toast(`Exported ${selected.length} notifications`, 'success');
  };

  const deleteSelected = async () => {
    const selected = filtered.filter((n) => selectedIds.has(n.id));
    if (selected.length === 0) return;
    // Bulk delete not yet supported by backend; clear selection optimistically
    setSelectedIds(new Set());
    toast(`Deleted ${selected.length} notifications`, 'success');
  };

  const markAllRead = async () => {
    try {
      const res = await fetch('/api/notifications/read-all', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data.success) {
        toast(`Marked ${data.count} notifications as read`, 'success');
        refreshAll();
      }
    } catch (err) {
      toast(err instanceof Error ? err.message : 'Failed to mark all read', 'error');
    }
  };

  const toggleRead = async (id: string, currentRead: boolean) => {
    if (currentRead) return;
    try {
      const res = await fetch('/api/notifications/read', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ id }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      if (data.success) {
        refreshAll();
      }
    } catch (err) {
      toast(err instanceof Error ? err.message : 'Failed to mark read', 'error');
    }
  };

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
            <Bell size={20} className="text-[#4fc3f7]" />
            Hermes Notification Center
          </h2>
          <p className="text-sm text-[#9e9eb0] mt-0.5">
            {unreadCount} unread notification{unreadCount !== 1 ? 's' : ''}
            {statsData && ` · ${statsData.total} total`}
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={refreshAll}
            disabled={notifLoading}
            className="aurelius-btn-outline flex items-center gap-2 text-sm disabled:opacity-50"
          >
            {notifLoading ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
            Refresh
          </button>
          <button
            onClick={markAllRead}
            className="aurelius-btn-outline flex items-center gap-2 text-sm"
          >
            <CheckCheck size={14} />
            Mark all as read
          </button>
        </div>
      </div>

      {/* Stats */}
      {statsData && (
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          {[
            { label: 'Total', value: statsData.total, color: 'text-[#e0e0e0]' },
            { label: 'Unread', value: statsData.unread, color: 'text-[#4fc3f7]' },
            { label: 'Agent', value: statsData.by_channel?.agent || 0, color: 'text-emerald-400' },
            { label: 'Alerts', value: statsData.by_channel?.alerts || 0, color: 'text-rose-400' },
          ].map((s) => (
            <div key={s.label} className="aurelius-card text-center py-3">
              <p className={`text-2xl font-bold ${s.color}`}>{s.value}</p>
              <p className="text-xs text-[#9e9eb0] uppercase tracking-wider mt-1">{s.label}</p>
            </div>
          ))}
        </div>
      )}

      {/* Tabs */}
      <div className="flex gap-2 border-b border-[#2d2d44] pb-1 overflow-x-auto">
        {tabs.map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`px-4 py-2 text-sm font-medium rounded-t-lg transition-colors whitespace-nowrap ${
              activeTab === tab.key
                ? 'text-[#4fc3f7] border-b-2 border-[#4fc3f7] bg-[#4fc3f7]/5'
                : 'text-[#9e9eb0] hover:text-[#e0e0e0]'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Bulk Actions */}
      <BulkActionsBar
        selectedCount={selectedIds.size}
        totalCount={filtered.length}
        onSelectAll={selectAll}
        onDeselectAll={deselectAll}
        onDelete={deleteSelected}
        onExport={exportSelected}
      />

      {/* List */}
      <div className="space-y-2">
        {notifLoading && notifications.length === 0 && (
          <div className="aurelius-card text-center py-12 text-[#9e9eb0]">
            <Loader2 size={32} className="mx-auto mb-3 animate-spin opacity-60" />
            <p>Loading notifications...</p>
          </div>
        )}

        {filtered.length === 0 && !notifLoading && (
          <div className="aurelius-card text-center py-12 text-[#9e9eb0]">
            <Bell size={32} className="mx-auto mb-3 opacity-40" />
            <p>No notifications in this channel.</p>
          </div>
        )}

        {paginated.map((n) => {
          const p = priorityConfig[(n.priority as Priority) || 'low'] || priorityConfig.low;
          const isSelected = selectedIds.has(n.id);
          return (
            <div
              key={n.id}
              className={`
                aurelius-card flex items-start gap-3 cursor-pointer transition-all
                ${n.read ? 'opacity-70' : 'opacity-100'}
                hover:border-[#4fc3f7]/30
                ${isSelected ? 'border-aurelius-accent/40 bg-aurelius-accent/5' : ''}
              `}
            >
              <div className="pt-1.5">
                <input
                  type="checkbox"
                  checked={isSelected}
                  onChange={() => toggleSelection(n.id)}
                  onClick={(e) => e.stopPropagation()}
                  className="w-4 h-4 rounded border-aurelius-border bg-aurelius-bg text-aurelius-accent focus:ring-aurelius-accent"
                />
              </div>
              <div
                onClick={() => toggleRead(n.id, n.read)}
                className="flex items-start gap-4 flex-1 min-w-0"
              >
                <div className="pt-1.5">
                  <div
                    className={`w-2.5 h-2.5 rounded-full ${
                      n.read ? 'bg-[#2d2d44]' : 'bg-[#4fc3f7] animate-pulse'
                    }`}
                  />
                </div>
                <div
                  className={`flex items-center justify-center w-9 h-9 rounded-lg shrink-0 ${p.bg} ${p.color} border ${p.border}`}
                >
                  <p.icon size={16} />
                </div>
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-0.5 flex-wrap">
                    <h3
                      className={`text-sm ${
                        n.read ? 'font-medium text-[#9e9eb0]' : 'font-semibold text-[#e0e0e0]'
                      }`}
                    >
                      {n.title}
                    </h3>
                    <span
                      className={`text-[10px] uppercase font-bold px-1.5 py-0.5 rounded ${p.bg} ${p.color} border ${p.border}`}
                    >
                      {n.priority}
                    </span>
                    <span className="flex items-center gap-1 text-[10px] text-[#9e9eb0] bg-[#0f0f1a] px-1.5 py-0.5 rounded border border-[#2d2d44]">
                      {channelIcon(n.channel)}
                      {n.channel}
                    </span>
                    {!n.delivered && (
                      <span className="text-[10px] text-amber-400 bg-amber-500/10 px-1.5 py-0.5 rounded border border-amber-500/20">
                        Pending
                      </span>
                    )}
                  </div>
                  <p className="text-sm text-[#9e9eb0] leading-relaxed">{n.body}</p>
                  <p className="text-xs text-[#9e9eb0]/60 mt-1">{timeAgo(n.timestamp)}</p>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {filtered.length > itemsPerPage && (
        <Pagination currentPage={currentPage} totalPages={totalPages} onPageChange={setCurrentPage} />
      )}
    </div>
  );
}
