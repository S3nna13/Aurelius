import { create } from 'zustand'
import { devtools } from 'zustand/middleware'
import { api } from '../services/api'

interface Workflow {
  id: string
  name: string
  status: 'running' | 'completed' | 'failed' | 'pending'
  steps: number
  currentStep: number
  startedAt: string
  duration?: string
}

interface WorkflowStoreState {
  workflows: Workflow[]
  loading: boolean
  error: string | null

  setWorkflows: (workflows: Workflow[]) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  updateStatus: (id: string, status: Workflow['status']) => void
  startWorkflow: (id: string) => Promise<void>
  stopWorkflow: (id: string) => void
  fetchWorkflows: () => Promise<void>
}

export const useWorkflowStore = create<WorkflowStoreState>()(
  devtools(
    (set) => ({
      workflows: [],
      loading: false,
      error: null,

      setWorkflows: (workflows) => set({ workflows }),
      setLoading: (loading) => set({ loading }),
      setError: (error) => set({ error }),

      updateStatus: (id, status) =>
        set((s) => ({
          workflows: s.workflows.map((w) => (w.id === id ? { ...w, status } : w)),
        })),

      startWorkflow: async (id) => {
        const res = await api.post<{ success?: boolean }>(`/api/workflows/${id}/start`, {})
        if (res.data?.success) {
          set((s) => ({
            workflows: s.workflows.map((w) =>
              w.id === id ? { ...w, status: 'running', currentStep: 1, startedAt: new Date().toISOString() } : w,
            ),
          }))
        }
      },

      stopWorkflow: async (id) => {
        await api.post(`/api/workflows/${id}/stop`)
        set((s) => ({
          workflows: s.workflows.map((w) => (w.id === id ? { ...w, status: 'pending', currentStep: 0 } : w)),
        }))
      },

      fetchWorkflows: async () => {
        set({ loading: true, error: null })
        try {
          const res = await api.get<{ workflows: Workflow[] }>('/api/status')
          if (res.data?.workflows) {
            set({ workflows: res.data.workflows, loading: false })
          }
        } catch (err) {
          set({ error: err instanceof Error ? err.message : 'Failed to load workflows', loading: false })
        }
      },
    }),
    { name: 'workflow-store' },
  ),
)
