import { useApiStore } from '../stores/apiStore'
import { useNotificationStore } from '../stores/notificationStore'
import { api } from './api'

interface HealthResponse {
  status: string
  version: string
  uptime: number
  memory: { rss: number; heapTotal: number; heapUsed: number }
  notifications: { unread: number; total: number }
}

interface NotificationResponse {
  notifications: Array<{
    id: string
    channel: string
    priority: string
    category: string
    title: string
    body: string
    read: boolean
    timestamp: number
  }>
}

export async function syncHealthToStore(): Promise<void> {
  const apiStore = useApiStore.getState()
  try {
    const { data } = await api.get<HealthResponse>('/health')
    if (data) {
      apiStore.setHealth({
        status: data.status,
        uptime: data.uptime,
        version: data.version,
        memory: data.memory,
      })
    }
  } catch {
    // silently fail
  }
}

export async function syncNotificationsToStore(): Promise<void> {
  const notifStore = useNotificationStore.getState()
  try {
    const { data } = await api.get<NotificationResponse>('/notifications')
    if (data?.notifications) {
      for (const n of data.notifications) {
        notifStore.addNotification({
          type: mapCategoryToType(n.category),
          title: n.title,
          message: n.body,
        })
      }
    }
  } catch {
    // silently fail
  }
}

function mapCategoryToType(category: string): 'info' | 'success' | 'warning' | 'error' {
  switch (category) {
    case 'startup': return 'info'
    case 'success': return 'success'
    case 'warning': return 'warning'
    case 'error': return 'error'
    default: return 'info'
  }
}

export async function syncAll(): Promise<void> {
  await Promise.allSettled([syncHealthToStore(), syncNotificationsToStore()])
}

export function startPeriodicSync(intervalMs = 15000): () => void {
  syncAll()
  const id = setInterval(syncAll, intervalMs)
  return () => clearInterval(id)
}
