import { useState } from 'react'
import { Bell, CheckCheck, AlertTriangle, Info, CheckCircle, XCircle } from 'lucide-react'
import { useNotificationStore } from '../stores/notificationStore'
import { Tabs } from './ui/Tabs'
import { EmptyState } from './EmptyState'
import { useWebSocket } from '../hooks/useWebSocket'
import { useEffect } from 'react'

export function NotificationFeed() {
  const { notifications, unreadCount, markRead, markAllRead, addNotification } = useNotificationStore()
  const ws = useWebSocket()
  const [filter, setFilter] = useState<string>('all')

  useEffect(() => {
    ws.on('notification', (payload) => {
      const p = payload as { title: string; body?: string; category?: string }
      addNotification({
        type: mapCategory(p.category || 'info'),
        title: p.title,
        message: p.body,
      })
    })
  }, [ws, addNotification])

  const filtered = filter === 'unread' ? notifications.filter((n) => !n.read) : notifications

  const tabs = [
    { id: 'all', label: 'All', badge: notifications.length },
    { id: 'unread', label: 'Unread', badge: unreadCount },
  ]

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <Tabs tabs={tabs} defaultTab="all" onChange={setFilter}>
          {() => null}
        </Tabs>
        {unreadCount > 0 && (
          <button onClick={markAllRead} className="text-[10px] text-[#4fc3f7] hover:underline flex items-center gap-1">
            <CheckCheck size={12} /> Mark all read
          </button>
        )}
      </div>

      {filtered.length === 0 ? (
        <EmptyState icon={<Bell size={24} />} title="No notifications" description={filter === 'unread' ? 'All caught up!' : 'No notifications yet'} />
      ) : (
        <div className="space-y-1 max-h-[400px] overflow-y-auto">
          {filtered.slice(0, 50).map((n) => (
            <div
              key={n.id}
              className={`flex items-start gap-3 px-3 py-2.5 rounded-lg cursor-pointer transition-colors ${n.read ? 'opacity-60 hover:opacity-80' : 'bg-[#4fc3f7]/5 hover:bg-[#4fc3f7]/10'}`}
              onClick={() => markRead(n.id)}
            >
              <NotificationIcon type={n.type} />
              <div className="flex-1 min-w-0">
                <p className={`text-sm ${n.read ? 'text-[#9e9eb0]' : 'text-[#e0e0e0]'}`}>{n.title}</p>
                {n.message && <p className="text-xs text-[#9e9eb0] truncate mt-0.5">{n.message}</p>}
              </div>
              <span className="text-[10px] text-[#9e9eb0]/40 shrink-0">{formatTime(n.timestamp)}</span>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function NotificationIcon({ type }: { type: string }) {
  const icons: Record<string, typeof Info> = { info: Info, success: CheckCircle, warning: AlertTriangle, error: XCircle }
  const colors: Record<string, string> = { info: 'text-[#4fc3f7]', success: 'text-emerald-400', warning: 'text-amber-400', error: 'text-rose-400' }
  const Icon = icons[type] || Info
  return <Icon size={14} className={`mt-0.5 shrink-0 ${colors[type] || 'text-[#9e9eb0]'}`} />
}

function mapCategory(cat: string): 'info' | 'success' | 'warning' | 'error' {
  if (cat === 'startup' || cat === 'info') return 'info'
  if (cat === 'success') return 'success'
  if (cat === 'warning') return 'warning'
  return 'error'
}

function formatTime(ts: number): string {
  const diff = Date.now() - ts
  if (diff < 60000) return 'now'
  if (diff < 3600000) return `${Math.floor(diff / 60000)}m`
  if (diff < 86400000) return `${Math.floor(diff / 3600000)}h`
  return `${Math.floor(diff / 86400000)}d`
}
