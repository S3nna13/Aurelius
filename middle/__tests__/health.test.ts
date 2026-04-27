import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import { buildApp } from '../src/server.js'
import type { Server } from 'http'

let server: Server
const BASE = 'http://127.0.0.1:3099'

beforeAll(async () => {
  const app = buildApp()
  server = app.listen(3099, '127.0.0.1')
  await new Promise((resolve) => server.on('listening', resolve))
})

afterAll(() => {
  server?.close()
})

describe('Health endpoints', () => {
  it('GET /health returns 200', async () => {
    const res = await fetch(`${BASE}/health`)
    expect(res.status).toBe(200)
    const data = await res.json()
    expect(data.status).toBe('ok')
  })

  it('GET /healthz returns alive', async () => {
    const res = await fetch(`${BASE}/healthz`)
    expect(res.status).toBe(200)
    const data = await res.json()
    expect(data.alive).toBe(true)
  })

  it('GET /readyz returns ready', async () => {
    const res = await fetch(`${BASE}/readyz`)
    expect(res.status).toBe(200)
    const data = await res.json()
    expect(data.ready).toBe(true)
  })
})

describe('Agent endpoints', () => {
  it('GET /api/agents returns list', async () => {
    const res = await fetch(`${BASE}/api/agents`, {
      headers: { 'X-API-Key': 'dev-key' },
    })
    expect(res.status).toBe(200)
    const data = await res.json()
    expect(data.agents).toBeDefined()
    expect(Array.isArray(data.agents)).toBe(true)
  })

  it('GET /api/agents returns agents with state', async () => {
    const res = await fetch(`${BASE}/api/agents`, {
      headers: { 'X-API-Key': 'dev-key' },
    })
    const data = await res.json()
    if (data.agents.length > 0) {
      expect(data.agents[0].id).toBeDefined()
      expect(data.agents[0].state).toBeDefined()
    }
  })
})

describe('Activity endpoints', () => {
  it('GET /api/activity returns entries', async () => {
    const res = await fetch(`${BASE}/api/activity`, {
      headers: { 'X-API-Key': 'dev-key' },
    })
    expect(res.status).toBe(200)
    const data = await res.json()
    expect(data.entries).toBeDefined()
  })
})

describe('Notification endpoints', () => {
  it('GET /api/notifications returns list', async () => {
    const res = await fetch(`${BASE}/api/notifications`, {
      headers: { 'X-API-Key': 'dev-key' },
    })
    expect(res.status).toBe(200)
    const data = await res.json()
    expect(data.notifications).toBeDefined()
  })

  it('GET /api/notifications/stats returns stats', async () => {
    const res = await fetch(`${BASE}/api/notifications/stats`, {
      headers: { 'X-API-Key': 'dev-key' },
    })
    expect(res.status).toBe(200)
    const data = await res.json()
    expect(data.unread).toBeDefined()
    expect(data.total).toBeDefined()
  })
})

describe('Config endpoints', () => {
  it('GET /api/config returns config', async () => {
    const res = await fetch(`${BASE}/api/config`, {
      headers: { 'X-API-Key': 'dev-key' },
    })
    expect(res.status).toBe(200)
    const data = await res.json()
    expect(data.config).toBeDefined()
  })
})

describe('Memory endpoints', () => {
  it('GET /api/memory/layers returns layers', async () => {
    const res = await fetch(`${BASE}/api/memory/layers`, {
      headers: { 'X-API-Key': 'dev-key' },
    })
    expect(res.status).toBe(200)
    const data = await res.json()
    expect(data.layers).toBeDefined()
  })
})

describe('Log endpoints', () => {
  it('GET /api/logs returns logs', async () => {
    const res = await fetch(`${BASE}/api/logs`, {
      headers: { 'X-API-Key': 'dev-key' },
    })
    expect(res.status).toBe(200)
    const data = await res.json()
    expect(data.logs).toBeDefined()
  })
})

describe('System endpoints', () => {
  it('GET /api/system returns system info', async () => {
    const res = await fetch(`${BASE}/api/system`, {
      headers: { 'X-API-Key': 'dev-key' },
    })
    expect(res.status).toBe(200)
    const data = await res.json()
    expect(data.hostname).toBeDefined()
    expect(data.cpu).toBeDefined()
    expect(data.memory).toBeDefined()
  })
})

describe('Stats endpoints', () => {
  it('GET /api/stats returns stats', async () => {
    const res = await fetch(`${BASE}/api/stats`, {
      headers: { 'X-API-Key': 'dev-key' },
    })
    expect(res.status).toBe(200)
    const data = await res.json()
    expect(data.agentCount).toBeDefined()
  })
})
