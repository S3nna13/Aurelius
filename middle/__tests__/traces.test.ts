import { describe, expect, it } from 'vitest'
import { buildApp } from '../src/server.js'
import { invokeApp } from './request-app.js'

const app = buildApp()

describe('Trace endpoints', () => {
  it('creates traces, appends steps, and only finalizes terminal statuses', async () => {
    const createRes = await invokeApp(app, {
      method: 'POST',
      path: '/api/traces',
      headers: {
        'X-API-Key': 'test-admin-key',
        'Content-Type': 'application/json',
      },
      body: {
        agentId: 'agent-123',
        agentName: 'Tracer',
        task: 'Run a trace slice',
      },
    })

    expect(createRes.status).toBe(201)
    const createData = await createRes.json()
    const traceId = createData.trace.id as string

    const listRes = await invokeApp(app, {
      method: 'GET',
      path: '/api/traces',
      headers: { 'X-API-Key': 'test-admin-key' },
    })
    expect(listRes.status).toBe(200)
    const listData = await listRes.json()
    expect(listData.total).toBeGreaterThanOrEqual(1)
    expect(listData.traces.some((trace: { id: string }) => trace.id === traceId)).toBe(true)

    const stepRes = await invokeApp(app, {
      method: 'POST',
      path: `/api/traces/${traceId}/step`,
      headers: {
        'X-API-Key': 'test-admin-key',
        'Content-Type': 'application/json',
      },
      body: {
        type: 'thought',
        content: 'Break the work into slices.',
        metadata: { source: 'test' },
        duration: 12,
      },
    })

    expect(stepRes.status).toBe(201)

    const runningRes = await invokeApp(app, {
      method: 'PATCH',
      path: `/api/traces/${traceId}`,
      headers: {
        'X-API-Key': 'test-admin-key',
        'Content-Type': 'application/json',
      },
      body: {
        status: 'running',
        tokenCount: 77,
      },
    })

    expect(runningRes.status).toBe(200)
    const runningData = await runningRes.json()
    expect(runningData.trace.status).toBe('running')
    expect(runningData.trace.tokenCount).toBe(77)
    expect(runningData.trace.completedAt).toBeUndefined()
    expect(runningData.trace.totalDuration).toBeUndefined()

    const completeRes = await invokeApp(app, {
      method: 'PATCH',
      path: `/api/traces/${traceId}`,
      headers: {
        'X-API-Key': 'test-admin-key',
        'Content-Type': 'application/json',
      },
      body: {
        status: 'completed',
      },
    })

    expect(completeRes.status).toBe(200)
    const completeData = await completeRes.json()
    expect(completeData.trace.status).toBe('completed')
    expect(completeData.trace.completedAt).toBeDefined()
    expect(completeData.trace.totalDuration).toBeGreaterThanOrEqual(0)

    const detailRes = await invokeApp(app, {
      method: 'GET',
      path: `/api/traces/${traceId}`,
      headers: { 'X-API-Key': 'test-admin-key' },
    })
    expect(detailRes.status).toBe(200)
    const detailData = await detailRes.json()
    expect(detailData.trace.steps).toHaveLength(1)
    expect(detailData.trace.steps[0].content).toBe('Break the work into slices.')
  })
})
