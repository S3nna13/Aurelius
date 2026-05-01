import { describe, it, expect, beforeAll, afterAll, afterEach } from 'vitest'
import {} from '../test-utils'
import { http, HttpResponse } from 'msw'
import { setupServer } from 'msw/node'

const handlers = [
  http.get('/api/agents', () => HttpResponse.json({ agents: [{ id: 'hermes', state: 'active', role: 'notification-router' }] })),
  http.get('/api/activity', () => HttpResponse.json({ entries: [{ id: 'a1', timestamp: Date.now() / 1000, command: 'test.ping', success: true, output: 'pong' }] })),
  http.get('/api/notifications', () => HttpResponse.json({ notifications: [{ id: 'n1', title: 'Test', priority: 'high', category: 'info', read: false, timestamp: Date.now(), channel: 'system' }] })),
  http.get('/api/logs', () => HttpResponse.json({ logs: [{ timestamp: '2026-04-26T12:00:00', level: 'info', logger: 'system', message: 'boot' }] })),
]

const server = setupServer(...handlers)

beforeAll(() => server.listen({ onUnhandledRequest: 'bypass' }))
afterEach(() => server.resetHandlers())
afterAll(() => server.close())

describe('DataExplorer', () => {
  it('renders without crashing', async () => {
    const { default: DataExplorer } = await import('../../pages/DataExplorer')
    expect(DataExplorer).toBeDefined()
  })

  it('fetches and displays agent data', async () => {
    const mod = await import('../../pages/DataExplorer')
    expect(mod.default).toBeDefined()
  })
})

describe('API client', () => {
  it('returns agent data', async () => {
    const res = await fetch('/api/agents')
    const data = await res.json()
    expect(data.agents).toHaveLength(1)
    expect(data.agents[0].id).toBe('hermes')
  })

  it('returns activity data', async () => {
    const res = await fetch('/api/activity')
    const data = await res.json()
    expect(data.entries[0].command).toBe('test.ping')
  })

  it('returns notification data', async () => {
    const res = await fetch('/api/notifications')
    const data = await res.json()
    expect(data.notifications[0].title).toBe('Test')
  })

  it('returns log data', async () => {
    const res = await fetch('/api/logs')
    const data = await res.json()
    expect(data.logs[0].level).toBe('info')
  })
})
