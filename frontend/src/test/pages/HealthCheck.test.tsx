import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '../test-utils'

describe('HealthCheck', () => {
  beforeEach(() => {
    localStorage.clear()
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('renders health status display', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(
      JSON.stringify({
        status: 'healthy',
        uptime: 86400,
        version: '1.0.0',
        services: [
          { name: 'Frontend', status: 'healthy', latency: 5 },
          { name: 'API (Node.js)', status: 'healthy', latency: 10 },
          { name: 'Model Serving (Python)', status: 'healthy', latency: 50 },
          { name: 'Database', status: 'healthy', latency: 8 },
        ],
        system: { cpu: 45, memory: 60, disk: 30 },
      }),
      { status: 200, headers: { 'Content-Type': 'application/json' } }
    )))
    const { default: HealthCheckPage } = await import('../../pages/HealthCheck')
    render(<HealthCheckPage />)

    await waitFor(() => {
      expect(screen.getByText('System Health')).toBeInTheDocument()
    }, { timeout: 10000 })
    expect(screen.getByText(/all systems operational/i)).toBeInTheDocument()
  })

  it('shows service list when health data loads', async () => {
    const mockFetch = vi.fn(async () => new Response(
      JSON.stringify({
        status: 'healthy',
        uptime: 86400,
        version: '1.0.0',
        services: [
          { name: 'Frontend', status: 'healthy', latency: 5 },
          { name: 'API (Node.js)', status: 'healthy', latency: 10 },
        ],
        system: { cpu: 45, memory: 60, disk: 30 },
      }),
      { status: 200, headers: { 'Content-Type': 'application/json' } }
    ))
    vi.stubGlobal('fetch', mockFetch)
    const { default: HealthCheckPage } = await import('../../pages/HealthCheck')
    render(<HealthCheckPage />)

    // Wait for services header to appear
    await waitFor(() => {
      expect(screen.getByText('Services')).toBeInTheDocument()
    }, { timeout: 15000 })
  })
})
