import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '../test-utils'
import userEvent from '@testing-library/user-event'

describe('Memory', () => {
  const mockMemoryData = {
    entries: [
      { id: '1', content: 'User prefers dark mode interface', type: 'preference', layer: 'user', timestamp: Date.now() - 86400000, tags: ['ui'] },
      { id: '2', content: 'Agent completed code review task', type: 'task', layer: 'session', timestamp: Date.now() - 3600000, tags: ['coding'] },
      { id: '3', content: 'Project deadline is next Friday', type: 'fact', layer: 'context', timestamp: Date.now() - 7200000, tags: ['schedule'] }
    ]
  }

  beforeEach(() => {
    localStorage.clear()
    vi.stubGlobal('fetch', vi.fn(async () => new Response(
      JSON.stringify(mockMemoryData),
      { status: 200, headers: { 'Content-Type': 'application/json' } }
    )))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('renders without crashing', async () => {
    const { default: Memory } = await import('../../pages/Memory')
    render(<Memory />)
    await waitFor(() => expect(screen.getByText('Memory')).toBeInTheDocument(), { timeout: 3000 })
  })

  it('displays memory entries', async () => {
    const { default: Memory } = await import('../../pages/Memory')
    render(<Memory />)
    await waitFor(() => expect(screen.getByText(/dark mode/i)).toBeInTheDocument(), { timeout: 3000 })
    expect(screen.getByText(/code review/i)).toBeInTheDocument()
    expect(screen.getAllByText(/deadline/i)[0]).toBeInTheDocument()
  })

  it('shows layer filter dropdown', async () => {
    const { default: Memory } = await import('../../pages/Memory')
    render(<Memory />)
    await waitFor(() => expect(screen.getByText('Memory')).toBeInTheDocument(), { timeout: 3000 })
    expect(screen.getByRole('combobox')).toBeInTheDocument()
  })

  it('shows search input', async () => {
    const { default: Memory } = await import('../../pages/Memory')
    render(<Memory />)
    await waitFor(() => expect(screen.getByText('Memory')).toBeInTheDocument(), { timeout: 3000 })
    expect(screen.getByPlaceholderText(/search memory/i)).toBeInTheDocument()
  })

  it('displays entry type badges', async () => {
    const { default: Memory } = await import('../../pages/Memory')
    render(<Memory />)
    await waitFor(() => expect(screen.getAllByText('preference').length).toBeGreaterThan(0), { timeout: 3000 })
    expect(screen.getAllByText('task').length).toBeGreaterThan(0)
    expect(screen.getAllByText('fact').length).toBeGreaterThan(0)
  })

  it('shows empty state when no entries', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(
      JSON.stringify({ entries: [] }),
      { status: 200, headers: { 'Content-Type': 'application/json' } }
    )))
    const { default: Memory } = await import('../../pages/Memory')
    render(<Memory />)
    await waitFor(() => expect(screen.getByText(/no memory entries/i)).toBeInTheDocument(), { timeout: 3000 })
  })
})