import { describe, expect, it } from 'vitest';
import { buildApp } from '../src/server.js';
import { invokeApp } from './request-app.js';

const app = buildApp();

describe('Plugins endpoints', () => {
  it('GET /api/plugins returns plugin list with expected shape', async () => {
    const res = await invokeApp(app, {
      method: 'GET',
      path: '/api/plugins',
      headers: { 'X-API-Key': 'test-admin-key' },
    });
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.plugins).toBeDefined();
    expect(Array.isArray(data.plugins)).toBe(true);
    expect(data.total).toBeDefined();
    expect(typeof data.total).toBe('number');
    expect(data.enabled).toBeDefined();
    expect(typeof data.enabled).toBe('number');
    expect(data.totalTools).toBeDefined();
    expect(data.totalSkills).toBeDefined();
  });

  it('GET /api/plugins returns plugin objects with id, name, and enabled fields', async () => {
    const res = await invokeApp(app, {
      method: 'GET',
      path: '/api/plugins',
      headers: { 'X-API-Key': 'test-admin-key' },
    });
    const data = await res.json();
    expect(data.plugins.length).toBeGreaterThan(0);
    const plugin = data.plugins[0];
    expect(plugin.id).toBeDefined();
    expect(plugin.name).toBeDefined();
    expect(plugin.enabled).toBeDefined();
    expect(typeof plugin.enabled).toBe('boolean');
  });

  it('GET /api/plugins/:id returns a specific plugin', async () => {
    const res = await invokeApp(app, {
      method: 'GET',
      path: '/api/plugins/filesystem',
      headers: { 'X-API-Key': 'test-admin-key' },
    });
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.plugin).toBeDefined();
    expect(data.plugin.id).toBe('filesystem');
    expect(data.plugin.name).toBe('Filesystem Tools');
  });

  it('GET /api/plugins/:id returns 404 for unknown plugin', async () => {
    const res = await invokeApp(app, {
      method: 'GET',
      path: '/api/plugins/nonexistent-plugin',
      headers: { 'X-API-Key': 'test-admin-key' },
    });
    expect(res.status).toBe(404);
    const data = await res.json();
    expect(data.error).toBe('Plugin not found');
  });

  it('PATCH /api/plugins/:id can disable a plugin', async () => {
    const res = await invokeApp(app, {
      method: 'PATCH',
      path: '/api/plugins/analytics',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': 'test-admin-key',
      },
      body: { enabled: false },
    });
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.plugin).toBeDefined();
    expect(data.plugin.enabled).toBe(false);
  });

  it('PATCH /api/plugins/:id can enable a plugin', async () => {
    const res = await invokeApp(app, {
      method: 'PATCH',
      path: '/api/plugins/database',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': 'test-admin-key',
      },
      body: { enabled: true },
    });
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.plugin).toBeDefined();
    expect(data.plugin.enabled).toBe(true);
  });
});
