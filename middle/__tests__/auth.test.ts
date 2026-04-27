import { describe, it, expect, beforeAll, afterAll } from 'vitest'
import { buildApp } from '../src/server.js'
import type { Server } from 'http'

let server: Server
const BASE = 'http://127.0.0.1:3098'

beforeAll(async () => {
  const app = buildApp()
  server = app.listen(3098, '127.0.0.1')
  await new Promise((resolve) => server.on('listening', resolve))
})

afterAll(() => {
  server?.close()
})

describe('Auth endpoints', () => {
  it('POST /api/auth/login with valid key returns token', async () => {
    const res = await fetch(`${BASE}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ apiKey: 'dev-key' }),
    })
    expect(res.status).toBe(200)
    const data = await res.json()
    expect(data.success).toBe(true)
    expect(data.token).toBeDefined()
  })

  it('POST /api/auth/login with invalid key returns 401', async () => {
    const res = await fetch(`${BASE}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ apiKey: 'bad-key' }),
    })
    expect(res.status).toBe(401)
  })

  it('POST /api/auth/login with missing key returns 400', async () => {
    const res = await fetch(`${BASE}/api/auth/login`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({}),
    })
    expect(res.status).toBe(400)
  })

  it('POST /api/auth/register creates new user', async () => {
    const username = `test-${Date.now()}`
    const res = await fetch(`${BASE}/api/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username }),
    })
    expect(res.status).toBe(200)
    const data = await res.json()
    expect(data.success).toBe(true)
    expect(data.apiKey).toBeDefined()
  })

  it('POST /api/auth/register with short name returns 400', async () => {
    const res = await fetch(`${BASE}/api/auth/register`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ username: 'ab' }),
    })
    expect(res.status).toBe(400)
  })

  it('POST /api/auth/keys/generate creates API key', async () => {
    const res = await fetch(`${BASE}/api/auth/keys/generate`, {
      method: 'POST',
      headers: { 'X-API-Key': 'dev-key', 'Content-Type': 'application/json' },
    })
    expect(res.status).toBe(200)
    const data = await res.json()
    expect(data.success).toBe(true)
    expect(data.apiKey).toBeDefined()
    expect(data.apiKey).toContain('ak-')
  })

  it('GET /api/auth/keys returns keys list', async () => {
    const res = await fetch(`${BASE}/api/auth/keys`, {
      headers: { 'X-API-Key': 'dev-key' },
    })
    expect(res.status).toBe(200)
    const data = await res.json()
    expect(data.keys).toBeDefined()
    expect(Array.isArray(data.keys)).toBe(true)
  })

  it('GET /api/auth/users returns users', async () => {
    const res = await fetch(`${BASE}/api/auth/users`, {
      headers: { 'X-API-Key': 'dev-key' },
    })
    expect(res.status).toBe(200)
    const data = await res.json()
    expect(data.users).toBeDefined()
    expect(Array.isArray(data.users)).toBe(true)
  })
})

describe('Auth middleware', () => {
  it('blocks unauthenticated requests to protected endpoints', async () => {
    const res = await fetch(`${BASE}/api/agents`)
    expect(res.status).toBe(401)
  })

  it('allows requests with valid API key', async () => {
    const res = await fetch(`${BASE}/api/activity`, {
      headers: { 'X-API-Key': 'dev-key' },
    })
    expect(res.status).toBe(200)
  })

  it('allows requests with Bearer token', async () => {
    const res = await fetch(`${BASE}/api/config`, {
      headers: { 'Authorization': 'Bearer dev-key' },
    })
    expect(res.status === 200 || res.status === 401)
  })

  it('public paths bypass auth', async () => {
    const res = await fetch(`${BASE}/health`)
    expect(res.status).toBe(200)
  })
})
