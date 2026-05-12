import { describe, expect, it } from 'vitest';
import { buildApp } from '../src/server.js';
import { invokeApp } from './request-app.js';

const app = buildApp();

describe('Search endpoints', () => {
  it('GET /api/search requires q parameter', async () => {
    const res = await invokeApp(app, {
      method: 'GET',
      path: '/api/search',
      headers: { 'X-API-Key': 'test-admin-key' },
    });
    expect(res.status).toBe(400);
    const data = await res.json();
    expect(data.error).toBe('Query parameter q is required');
  });

  it('GET /api/search?q=... returns search results with expected shape', async () => {
    const res = await invokeApp(app, {
      method: 'GET',
      path: '/api/search?q=system',
      headers: { 'X-API-Key': 'test-admin-key' },
    });
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.query).toBe('system');
    expect(data.type).toBe('all');
    expect(data.results).toBeDefined();
    expect(typeof data.results).toBe('object');
    expect(data.total).toBeDefined();
    expect(typeof data.total).toBe('number');
  });

  it('GET /api/search?q=... returns results object with category keys', async () => {
    const res = await invokeApp(app, {
      method: 'GET',
      path: '/api/search?q=agent',
      headers: { 'X-API-Key': 'test-admin-key' },
    });
    const data = await res.json();
    // Results should contain keys for different entity types
    expect(data.results).toHaveProperty('agents');
    expect(data.results).toHaveProperty('activity');
    expect(data.results).toHaveProperty('logs');
  });

  it('GET /api/search?q=...&type=agents filters to agents only', async () => {
    const res = await invokeApp(app, {
      method: 'GET',
      path: '/api/search?q=hermes&type=agents',
      headers: { 'X-API-Key': 'test-admin-key' },
    });
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.type).toBe('agents');
    expect(data.results.agents).toBeDefined();
  });

  it('GET /api/search/suggestions returns suggestions array', async () => {
    const res = await invokeApp(app, {
      method: 'GET',
      path: '/api/search/suggestions',
      headers: { 'X-API-Key': 'test-admin-key' },
    });
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.suggestions).toBeDefined();
    expect(Array.isArray(data.suggestions)).toBe(true);
    expect(data.suggestions.length).toBeGreaterThan(0);
  });
});
