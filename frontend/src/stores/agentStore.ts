import { create } from 'zustand'
import type { AgentState } from '../api/types'

interface AgentStoreState {
  agents: AgentState[]
  selectedId: string | null
  connected: boolean
  setAgents: (agents: AgentState[]) => void
  updateAgent: (agent: AgentState) => void
  setSelected: (id: string | null) => void
  setConnected: (connected: boolean) => void
}

export const useAgentStore = create<AgentStoreState>()(
  (set) => ({
    agents: [],
    selectedId: null,
    connected: false,

    setAgents: (agents) => set({ agents }),
    updateAgent: (agent) =>
      set((s) => ({
        agents: s.agents.map((a) => (a.id === agent.id ? agent : a)),
      })),
    setSelected: (selectedId) => set({ selectedId }),
    setConnected: (connected) => set({ connected }),
  }),
)
