import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '../test-utils'
import userEvent from '@testing-library/user-event'

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom')
  return {
    ...actual,
    useParams: () => ({ id: '1' }),
    useNavigate: () => vi.fn()
  }
})

describe('AgentDetail', () => {
  beforeEach(() => {
    localStorage.clear()
    vi.stubGlobal('fetch', vi.fn(async (input: RequestInfo | URL) => {
      const url = typeof input === 'string' ? input : input instanceof URL ? input.href : (input as Request).url
      if (url.includes('/api/agents/')) {
        return new Response(
          JSON.stringify({
            agent: {
              id: '1',
              name: 'Test Agent',
              role: 'coding',
              state: 'active',
              capabilities: ['read_file', 'write_file', 'search_web'],
              created: Date.now(),
              metrics: {
                tasksCompleted: 42,
                avgLatencyMs: 150.5,
                errorRate: 0.02,
                tokensUsed: 50000,
                uptimeHours: 24.5
              }
            }
          }),
          { status: 200, headers: { 'Content-Type': 'application/json' } }
        )
      }
      return new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } })
    }))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('renders without crashing', async () => {
    const { default: AgentDetail } = await import('../../pages/AgentDetail')
    render(<AgentDetail />)
    await waitFor(() => expect(screen.queryByText('Test Agent')).toBeInTheDocument(), { timeout: 3000 })
  })

  it('displays agent metrics', async () => {
    const { default: AgentDetail } = await import('../../pages/AgentDetail')
    render(<AgentDetail />)
    await waitFor(() => expect(screen.getByText('42')).toBeInTheDocument(), { timeout: 3000 })
    await waitFor(() => expect(screen.getByText('151ms')).toBeInTheDocument(), { timeout: 3000 })
    await waitFor(() => expect(screen.getByText('2.0%')).toBeInTheDocument(), { timeout: 3000 })
  })

  it('displays agent capabilities', async () => {
    const { default: AgentDetail } = await import('../../pages/AgentDetail')
    render(<AgentDetail />)
    await waitFor(() => expect(screen.getAllByText('read_file').length).toBeGreaterThan(0), { timeout: 3000 })
    expect(screen.getAllByText('write_file').length).toBeGreaterThan(0)
    expect(screen.getAllByText('search_web').length).toBeGreaterThan(0)
  })

  it('shows agent state badge', async () => {
    const { default: AgentDetail } = await import('../../pages/AgentDetail')
    render(<AgentDetail />)
    await waitFor(() => expect(screen.getByText('active')).toBeInTheDocument(), { timeout: 3000 })
  })

  it('shows back button', async () => {
    const { default: AgentDetail } = await import('../../pages/AgentDetail')
    render(<AgentDetail />)
    await waitFor(() => expect(screen.queryByText('Test Agent')).toBeInTheDocument(), { timeout: 3000 })
    expect(screen.getByRole('button')).toBeInTheDocument()
  })
})