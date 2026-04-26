import { create } from 'zustand'
import { devtools } from 'zustand/middleware'

export interface Notification {
  id: string
  type: 'info' | 'success' | 'warning' | 'error'
  title: string
  message?: string
  timestamp: number
  read: boolean
}

interface NotificationState {
  notifications: Notification[]
  unreadCount: number

  addNotification: (n: Omit<Notification, 'id' | 'timestamp' | 'read'>) => void
  markRead: (id: string) => void
  markAllRead: () => void
  dismiss: (id: string) => void
  clear: () => void
}

let _notifId = 0

export const useNotificationStore = create<NotificationState>()(
  devtools(
    (set) => ({
      notifications: [],
      unreadCount: 0,

      addNotification: (n) =>
        set((s) => {
          const notif: Notification = {
            ...n,
            id: `notif-${++_notifId}`,
            timestamp: Date.now(),
            read: false,
          }
          return {
            notifications: [notif, ...s.notifications].slice(0, 100),
            unreadCount: s.unreadCount + 1,
          }
        }),

      markRead: (id) =>
        set((s) => {
          const notifications = s.notifications.map((n) =>
            n.id === id ? { ...n, read: true } : n
          )
          return {
            notifications,
            unreadCount: notifications.filter((n) => !n.read).length,
          }
        }),

      markAllRead: () =>
        set((s) => ({
          notifications: s.notifications.map((n) => ({ ...n, read: true })),
          unreadCount: 0,
        })),

      dismiss: (id) =>
        set((s) => {
          const notifications = s.notifications.filter((n) => n.id !== id)
          return {
            notifications,
            unreadCount: notifications.filter((n) => !n.read).length,
          }
        }),

      clear: () => set({ notifications: [], unreadCount: 0 }),
    }),
    { name: 'notification-store' }
  )
)
