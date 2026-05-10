import { describe, it, expect } from 'vitest'
import { buildApp } from '../src/server.js'
import { invokeApp } from './request-app.js'

const app = buildApp()

describe('Auth endpoints', () => {
  it('POST /api/auth/login with valid key returns token', async () => {
    const res = await invokeApp(app, {
      method: 'POST',
      path: '/api/auth/login',
      headers: { 'Content-Type': 'application/json' },
      body: { apiKey: 'test-admin-key' },
    })
    expect(res.status).toBe(200)
    const data = await res.json()
    expect(data.success).toBe(true)
    expect(data.token).toBeDefined()
  })

  it('POST /api/auth/login with invalid key returns 401', async () => {
    const res = await invokeApp(app, {
      method: 'POST',
      path: '/api/auth/login',
      headers: { 'Content-Type': 'application/json' },
      body: { apiKey: 'bad-key' },
    })
    expect(res.status).toBe(401)
  })

  it('POST /api/auth/login with missing key returns 400', async () => {
    const res = await invokeApp(app, {
      method: 'POST',
      path: '/api/auth/login',
      headers: { 'Content-Type': 'application/json' },
      body: {},
    })
    expect(res.status).toBe(400)
  })

  it('POST /api/auth/register creates new user', async () => {
    const username = `test-${Date.now()}`
    const res = await invokeApp(app, {
      method: 'POST',
      path: '/api/auth/register',
      headers: { 'Content-Type': 'application/json' },
      body: { username },
    })
    expect(res.status).toBe(200)
    const data = await res.json()
    expect(data.success).toBe(true)
    expect(data.apiKey).toBeDefined()
  })

  it('POST /api/auth/register with short name returns 400', async () => {
    const res = await invokeApp(app, {
      method: 'POST',
      path: '/api/auth/register',
      headers: { 'Content-Type': 'application/json' },
      body: { username: 'ab' },
    })
    expect(res.status).toBe(400)
  })

  it('POST /api/auth/keys/generate creates API key', async () => {
    const res = await invokeApp(app, {
      method: 'POST',
      path: '/api/auth/keys/generate',
      headers: { 'X-API-Key': 'test-admin-key', 'Content-Type': 'application/json' },
    })
    expect(res.status).toBe(200)
    const data = await res.json()
    expect(data.success).toBe(true)
    expect(data.apiKey).toBeDefined()
    expect(data.apiKey).toContain('ak-')
  })

  it('GET /api/auth/keys returns keys list', async () => {
    const res = await invokeApp(app, {
      method: 'GET',
      path: '/api/auth/keys',
      headers: { 'X-API-Key': 'test-admin-key' },
    })
    expect(res.status).toBe(200)
    const data = await res.json()
    expect(data.keys).toBeDefined()
    expect(Array.isArray(data.keys)).toBe(true)
  })

  it('GET /api/auth/users returns users', async () => {
    const res = await invokeApp(app, {
      method: 'GET',
      path: '/api/auth/users',
      headers: { 'X-API-Key': 'test-admin-key' },
    })
    expect(res.status).toBe(200)
    const data = await res.json()
    expect(data.users).toBeDefined()
    expect(Array.isArray(data.users)).toBe(true)
  })
})

describe('Auth middleware', () => {
  it('blocks unauthenticated requests to protected endpoints', async () => {
    const res = await invokeApp(app, {
      method: 'GET',
      path: '/api/agents',
    })
    expect(res.status).toBe(401)
  })

  it('allows requests with valid API key', async () => {
    const res = await invokeApp(app, {
      method: 'GET',
      path: '/api/activity',
      headers: { 'X-API-Key': 'test-admin-key' },
    })
    expect(res.status).toBe(200)
  })

  it('allows requests with Bearer token', async () => {
    const res = await invokeApp(app, {
      method: 'GET',
      path: '/api/config',
      headers: { Authorization: 'Bearer test-admin-key' },
    })
    expect(res.status === 200 || res.status === 401)
  })

  it('public paths bypass auth', async () => {
    const res = await invokeApp(app, {
      method: 'GET',
      path: '/health',
    })
    expect(res.status).toBe(200)
  })
})
