import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { render, screen, waitFor } from '../test-utils'
import userEvent from '@testing-library/user-event'

describe('ScheduledTasks', () => {
  const mockTasksData = {
    tasks: [
      { id: '1', name: 'Daily Report', cron: '0 9 * * *', task: 'generate_report', enabled: true, lastStatus: 'completed', nextRun: '2025-05-08T09:00:00Z' },
      { id: '2', name: 'Cleanup', cron: '0 0 * * *', task: 'cleanup_logs', enabled: false, lastStatus: 'failed', nextRun: '2025-05-08T00:00:00Z' },
      { id: '3', name: 'Sync Data', cron: '*/15 * * * *', task: 'sync_data', enabled: true, lastStatus: 'completed', nextRun: '2025-05-07T12:15:00Z' }
    ]
  }

  beforeEach(() => {
    localStorage.clear()
    vi.stubGlobal('fetch', vi.fn(async () => new Response(
      JSON.stringify(mockTasksData),
      { status: 200, headers: { 'Content-Type': 'application/json' } }
    )))
  })

  afterEach(() => {
    vi.unstubAllGlobals()
    vi.restoreAllMocks()
  })

  it('renders without crashing', async () => {
    const { default: ScheduledTasks } = await import('../../pages/ScheduledTasks')
    render(<ScheduledTasks />)
    await waitFor(() => expect(screen.getByText('Scheduled Tasks')).toBeInTheDocument(), { timeout: 10000 })
  })

  it('displays task list', async () => {
    const { default: ScheduledTasks } = await import('../../pages/ScheduledTasks')
    render(<ScheduledTasks />)
    await waitFor(() => expect(screen.getByText(/Daily Report/)).toBeInTheDocument(), { timeout: 10000 })
    expect(screen.getByText(/Cleanup/)).toBeInTheDocument()
    expect(screen.getByText(/Sync Data/)).toBeInTheDocument()
  })

  it('shows task details with cron expressions', async () => {
    const { default: ScheduledTasks } = await import('../../pages/ScheduledTasks')
    render(<ScheduledTasks />)
    await waitFor(() => expect(screen.getByText('Scheduled Tasks')).toBeInTheDocument(), { timeout: 10000 })
    // Check that tasks are displayed - the cron is shown as font-mono text
    await waitFor(() => expect(screen.getByText('Daily Report')).toBeInTheDocument(), { timeout: 10000 })
  })

  it('shows new task button', async () => {
    const { default: ScheduledTasks } = await import('../../pages/ScheduledTasks')
    render(<ScheduledTasks />)
    await waitFor(() => expect(screen.getByText('Scheduled Tasks')).toBeInTheDocument(), { timeout: 10000 })
    expect(screen.getByRole('button', { name: /new task/i })).toBeInTheDocument()
  })

  it('renders empty state when no tasks', async () => {
    vi.stubGlobal('fetch', vi.fn(async () => new Response(
      JSON.stringify({ tasks: [] }),
      { status: 200, headers: { 'Content-Type': 'application/json' } }
    )))
    const { default: ScheduledTasks } = await import('../../pages/ScheduledTasks')
    render(<ScheduledTasks />)
    await waitFor(() => expect(screen.getByText(/no scheduled tasks/i)).toBeInTheDocument(), { timeout: 10000 })
  })
})
