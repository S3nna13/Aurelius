import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '../test-utils'
import userEvent from '@testing-library/user-event'

describe('Notifications', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  const mockFetch = vi.fn(async (input: RequestInfo | URL) => {
    const url = typeof input === 'string' ? input : input instanceof URL ? input.href : (input as Request).url

    if (url.includes('/notifications/stats')) {
      return new Response(
        JSON.stringify({ unread: 1, total: 2, by_channel: { agent: 1, system: 1 }, by_priority: { high: 1, low: 1 } }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      )
    }

    if (url.includes('/notifications')) {
      return new Response(
        JSON.stringify({
          notifications: [
            { id: '1', title: 'Test Notification', body: 'Test body', channel: 'agent', priority: 'high', read: false, delivered: true, timestamp: Date.now() / 1000 },
            { id: '2', title: 'Another Notification', body: 'Another body', channel: 'system', priority: 'low', read: true, delivered: true, timestamp: Date.now() / 1000 - 100 },
          ],
        }),
        { status: 200, headers: { 'Content-Type': 'application/json' } }
      )
    }

    return new Response('{}', { status: 200, headers: { 'Content-Type': 'application/json' } })
  })

  it('renders notification list', async () => {
    vi.stubGlobal('fetch', mockFetch)
    const { default: Notifications } = await import('../../pages/Notifications')
    render(<Notifications />)

    await waitFor(() => {
      expect(screen.getByText('Hermes Notification Center')).toBeInTheDocument()
    }, { timeout: 10000 })
  })

  it('has filter/tab controls', async () => {
    vi.stubGlobal('fetch', mockFetch)
    const { default: Notifications } = await import('../../pages/Notifications')
    render(<Notifications />)

    await waitFor(() => {
      expect(screen.getByText('All')).toBeInTheDocument()
      expect(screen.getByText('Agent')).toBeInTheDocument()
      expect(screen.getByText('System')).toBeInTheDocument()
      expect(screen.getByText('Alerts')).toBeInTheDocument()
    }, { timeout: 10000 })
  })

  it('has mark-all-read button', async () => {
    vi.stubGlobal('fetch', mockFetch)
    const { default: Notifications } = await import('../../pages/Notifications')
    render(<Notifications />)

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /mark all as read/i })).toBeInTheDocument()
    }, { timeout: 10000 })
  })
})
