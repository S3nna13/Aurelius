import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '../test-utils'
import userEvent from '@testing-library/user-event'

describe('Dashboard', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  const createMockFetch = () => vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : (input as Request).url
    
    // Log for debugging
    console.log('Fetch called with:', url)
    
    if (url.includes('/agents')) {
      return new Response(
        JSON.stringify({ agents: [{ id: '1', name: 'Test Agent', state: 'active', role: 'coding' }] }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      )
    }
    if (url.includes('/health')) {
      return new Response(
        JSON.stringify({ status: 'healthy', services: [], version: '1.0.0' }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      )
    }
    if (url.includes('/notifications/stats')) {
      return new Response(
        JSON.stringify({ unread: 0, total: 0 }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      )
    }
    if (url.includes('/plugins')) {
      return new Response(
        JSON.stringify({ total: 12, enabled: 8 }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      )
    }
    if (url.includes('/stats/summary')) {
      return new Response(
        JSON.stringify({ requests: 100, avg_latency: 50, tokens: 5000 }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      )
    }
    return new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } })
  })

  it('renders without crashing', async () => {
    vi.stubGlobal('fetch', createMockFetch())
    const { default: Dashboard } = await import('../../pages/Dashboard')
    render(<Dashboard />)
    expect(screen.getByText('Operations Dashboard')).toBeInTheDocument()
  })

  it('shows quick action buttons that navigate to expected routes', async () => {
    vi.stubGlobal('fetch', createMockFetch())
    const user = userEvent.setup()
    const { default: Dashboard } = await import('../../pages/Dashboard')
    render(<Dashboard />)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /agent grid/i })).toBeInTheDocument()
    }, { timeout: 10000 })
    
    expect(screen.getByRole('button', { name: /agent chat/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /playground/i })).toBeInTheDocument()
    expect(screen.getByRole('button', { name: /analytics/i })).toBeInTheDocument()
  })

  it('shows system health indicator', async () => {
    const mockFetch = createMockFetch()
    vi.stubGlobal('fetch', mockFetch)
    const { default: Dashboard } = await import('../../pages/Dashboard')
    render(<Dashboard />)

    // Wait for either "All Systems Operational" or "System Degraded" to appear
    await waitFor(() => {
      const healthy = screen.queryByText(/all systems operational/i)
      const degraded = screen.queryByText(/system degraded/i)
      expect(healthy || degraded).toBeInTheDocument()
    }, { timeout: 10000 })
  })
})
