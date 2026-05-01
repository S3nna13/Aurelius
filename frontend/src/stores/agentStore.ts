import { create } from 'zustand'
import { devtools } from 'zustand/middleware'

export interface Agent {
  id: string
  state: string
  role: string
  metrics_json: string
}

interface AgentStoreState {
  agents: Agent[]
  loading: boolean
  error: string | null
  selectedId: string | null

  setAgents: (agents: Agent[]) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  selectAgent: (id: string | null) => void
  updateAgentState: (id: string, state: string) => void
  fetchAgents: () => Promise<void>
}

export const useAgentStore = create<AgentStoreState>()(
  devtools(
    (set, _get) => ({
      agents: [],
      loading: false,
      error: null,
      selectedId: null,

      setAgents: (agents) => set({ agents }),
      setLoading: (loading) => set({ loading }),
      setError: (error) => set({ error }),
      selectAgent: (id) => set({ selectedId: id }),

      updateAgentState: (id, state) =>
        set((s) => ({
          agents: s.agents.map((a) => (a.id === id ? { ...a, state } : a)),
        })),

      fetchAgents: async () => {
        set({ loading: true, error: null })
        try {
          const res = await fetch('/api/agents')
          if (!res.ok) throw new Error(`HTTP ${res.status}`)
          const data = await res.json()
          set({ agents: data.agents || [], loading: false })
        } catch (err) {
          set({ error: err instanceof Error ? err.message : 'Failed to load agents', loading: false })
        }
      },
    }),
    { name: 'agent-store' },
  ),
)
