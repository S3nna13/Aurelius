import { afterAll, beforeAll, describe, expect, it } from 'vitest'
import { buildApp } from '../src/server.js'
import type { Server } from 'http'

let server: Server
let baseUrl = ''

beforeAll(async () => {
  const app = buildApp()
  server = app.listen(0, '127.0.0.1')
  await new Promise((resolve) => server.on('listening', resolve))
  const address = server.address()
  if (address && typeof address === 'object') {
    baseUrl = `http://127.0.0.1:${address.port}`
  } else {
    throw new Error('Failed to determine test server port')
  }
})

afterAll(() => {
  server?.close()
})

describe('Trace endpoints', () => {
  it('creates traces, appends steps, and only finalizes terminal statuses', async () => {
    const createRes = await fetch(`${baseUrl}/api/traces`, {
      method: 'POST',
      headers: {
        'X-API-Key': 'test-admin-key',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        agentId: 'agent-123',
        agentName: 'Tracer',
        task: 'Run a trace slice',
      }),
    })

    expect(createRes.status).toBe(201)
    const createData = await createRes.json()
    const traceId = createData.trace.id as string

    const listRes = await fetch(`${baseUrl}/api/traces`, {
      headers: { 'X-API-Key': 'test-admin-key' },
    })
    expect(listRes.status).toBe(200)
    const listData = await listRes.json()
    expect(listData.total).toBeGreaterThanOrEqual(1)
    expect(listData.traces.some((trace: { id: string }) => trace.id === traceId)).toBe(true)

    const stepRes = await fetch(`${baseUrl}/api/traces/${traceId}/step`, {
      method: 'POST',
      headers: {
        'X-API-Key': 'test-admin-key',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        type: 'thought',
        content: 'Break the work into slices.',
        metadata: { source: 'test' },
        duration: 12,
      }),
    })

    expect(stepRes.status).toBe(201)

    const runningRes = await fetch(`${baseUrl}/api/traces/${traceId}`, {
      method: 'PATCH',
      headers: {
        'X-API-Key': 'test-admin-key',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        status: 'running',
        tokenCount: 77,
      }),
    })

    expect(runningRes.status).toBe(200)
    const runningData = await runningRes.json()
    expect(runningData.trace.status).toBe('running')
    expect(runningData.trace.tokenCount).toBe(77)
    expect(runningData.trace.completedAt).toBeUndefined()
    expect(runningData.trace.totalDuration).toBeUndefined()

    const completeRes = await fetch(`${baseUrl}/api/traces/${traceId}`, {
      method: 'PATCH',
      headers: {
        'X-API-Key': 'test-admin-key',
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        status: 'completed',
      }),
    })

    expect(completeRes.status).toBe(200)
    const completeData = await completeRes.json()
    expect(completeData.trace.status).toBe('completed')
    expect(completeData.trace.completedAt).toBeDefined()
    expect(completeData.trace.totalDuration).toBeGreaterThanOrEqual(0)

    const detailRes = await fetch(`${baseUrl}/api/traces/${traceId}`, {
      headers: { 'X-API-Key': 'test-admin-key' },
    })
    expect(detailRes.status).toBe(200)
    const detailData = await detailRes.json()
    expect(detailData.trace.steps).toHaveLength(1)
    expect(detailData.trace.steps[0].content).toBe('Break the work into slices.')
  })
})
