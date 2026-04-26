import { create } from 'zustand'
import { devtools } from 'zustand/middleware'

interface AppState {
  sidebarOpen: boolean
  searchOpen: boolean
  paletteOpen: boolean
  autoRefresh: boolean

  toggleSidebar: () => void
  setSearchOpen: (open: boolean) => void
  setPaletteOpen: (open: boolean) => void
  setAutoRefresh: (enabled: boolean) => void
}

export const useAppStore = create<AppState>()(
  devtools(
    (set) => ({
      sidebarOpen: true,
      searchOpen: false,
      paletteOpen: false,
      autoRefresh: localStorage.getItem('aurelius-auto-refresh') !== 'false',

      toggleSidebar: () => set((s) => ({ sidebarOpen: !s.sidebarOpen })),
      setSearchOpen: (open) => set({ searchOpen: open }),
      setPaletteOpen: (open) => set({ paletteOpen: open }),
      setAutoRefresh: (enabled) => {
        localStorage.setItem('aurelius-auto-refresh', String(enabled))
        set({ autoRefresh: enabled })
      },
    }),
    { name: 'app-store' }
  )
)
