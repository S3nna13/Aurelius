/**
 * Zustand persistence middleware that saves/restores store state
 * to localStorage with automatic migration support.
 */

import { create, type StateCreator } from 'zustand'
import { persist, createJSONStorage } from 'zustand/middleware'

type PersistConfig<T> = {
  name: string
  version?: number
  partialize?: (state: T) => Partial<T>
  migrate?: (persisted: unknown, version: number) => T
  onRehydrateStorage?: () => void
}

export function createPersistedStore<T extends object>(
  initializer: StateCreator<T, [], []>,
  config: PersistConfig<T>,
) {
  return create<T>()(
    persist(initializer, {
      name: `aurelius-${config.name}`,
      version: config.version ?? 1,
      storage: createJSONStorage(() => localStorage),
      partialize: config.partialize,
      migrate: config.migrate as (persisted: unknown, version: number) => T,
      onRehydrateStorage: config.onRehydrateStorage,
    }) as unknown as StateCreator<T, [], []>,
  )
}

const migrations: Record<string, (state: Record<string, unknown>, version: number) => Record<string, unknown>> = {
  'aurelius-app-store': (state, version) => {
    if (version < 2) {
      return { ...state, autoRefresh: true, paletteOpen: false }
    }
    return state
  },
}

export function migrateStore(storeName: string, state: unknown, version: number): unknown {
  const migration = migrations[storeName]
  if (migration) {
    return migration(state as Record<string, unknown>, version)
  }
  return state
}
