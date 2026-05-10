import { afterAll, beforeAll, describe, expect, it, vi } from 'vitest';
import { buildApp } from '../src/server.js';
import { invokeApp } from './request-app.js';

let app: ReturnType<typeof buildApp>;

const mockModelsResponse = {
  object: 'list',
  data: [
    { id: 'aurelius-1.3b', object: 'model', created: 1714300000, owned_by: 'aurelius' },
    { id: 'aurelius-2.7b', object: 'model', created: 1714300001, owned_by: 'aurelius' },
  ],
};

beforeAll(async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response(JSON.stringify(mockModelsResponse), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    })),
  );

  app = buildApp();
}, 30000);

afterAll(async () => {
  vi.unstubAllGlobals();
});

describe('Models endpoints', () => {
  it('GET /api/v1/models returns available models', async () => {
    const res = await invokeApp(app, {
      method: 'GET',
      path: '/api/models',
      headers: { 'X-API-Key': 'test-admin-key' },
    });
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data).toBeDefined();
    // The upstream returns an object with a data array
    expect(data.data).toBeDefined();
    expect(Array.isArray(data.data)).toBe(true);
  });

  it('GET /api/v1/models returns model objects with id field', async () => {
    const res = await invokeApp(app, {
      method: 'GET',
      path: '/api/models',
      headers: { 'X-API-Key': 'test-admin-key' },
    });
    const data = await res.json();
    expect(data.data.length).toBeGreaterThan(0);
    expect(data.data[0].id).toBeDefined();
  });
});
