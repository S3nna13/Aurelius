import { create } from 'zustand'
import { devtools } from 'zustand/middleware'

interface HealthStatus {
  status: string
  uptime: number
  version: string
  memory: { rss: number }
}

interface ModelInfo {
  id: string
  object: string
  created: number
  owned_by: string
}

interface ApiStoreState {
  baseUrl: string
  apiKey: string
  health: HealthStatus | null
  models: ModelInfo[]
  loading: boolean
  error: string | null

  setBaseUrl: (url: string) => void
  setApiKey: (key: string) => void
  clearApiKey: () => void
  setHealth: (health: HealthStatus) => void
  setModels: (models: ModelInfo[]) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
}

export const useApiStore = create<ApiStoreState>()(
  devtools(
    (set) => ({
      baseUrl: import.meta.env.VITE_API_BASE_URL || '/api',
      apiKey: localStorage.getItem('aurelius-api-key') || '',
      health: null,
      models: [],
      loading: false,
      error: null,

      setBaseUrl: (url) => set({ baseUrl: url }),
      setApiKey: (key) => {
        localStorage.setItem('aurelius-api-key', key)
        set({ apiKey: key })
      },
      clearApiKey: () => {
        localStorage.removeItem('aurelius-api-key')
        set({ apiKey: '' })
      },
      setHealth: (health) => set({ health }),
      setModels: (models) => set({ models }),
      setLoading: (loading) => set({ loading }),
      setError: (error) => set({ error }),
    }),
    { name: 'api-store' }
  )
)
