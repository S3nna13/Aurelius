import { describe, expect, it } from 'vitest';
import { buildApp } from '../src/server.js';
import { invokeApp } from './request-app.js';

const app = buildApp();

describe('Scheduler endpoints', () => {
  it('GET /api/scheduler returns tasks array', async () => {
    const res = await invokeApp(app, {
      method: 'GET',
      path: '/api/scheduler',
      headers: { 'X-API-Key': 'test-admin-key' },
    });
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.tasks).toBeDefined();
    expect(Array.isArray(data.tasks)).toBe(true);
  });

  it('POST /api/scheduler creates a new task', async () => {
    const res = await invokeApp(app, {
      method: 'POST',
      path: '/api/scheduler',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': 'test-admin-key',
      },
      body: {
        name: 'Test Task',
        cron: '5 * * * *',
        command: 'echo "hello"',
      },
    });
    expect(res.status).toBe(200);
    const data = await res.json();
    expect(data.success).toBe(true);
    expect(data.task).toBeDefined();
    expect(data.task.name).toBe('Test Task');
    expect(data.task.cron).toBe('5 * * * *');
    expect(data.task.enabled).toBe(true);
  });

  it('POST /api/scheduler returns 400 when missing required fields', async () => {
    const res = await invokeApp(app, {
      method: 'POST',
      path: '/api/scheduler',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': 'test-admin-key',
      },
      body: { name: 'Incomplete Task' },
    });
    expect(res.status).toBe(400);
    const data = await res.json();
    expect(data.error).toBeDefined();
  });

  it('DELETE /api/scheduler/:id removes a task', async () => {
    // First create a task
    const createRes = await invokeApp(app, {
      method: 'POST',
      path: '/api/scheduler',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': 'test-admin-key',
      },
      body: {
        name: 'Task to Delete',
        cron: '10 * * * *',
        command: 'echo "delete me"',
      },
    });
    const created = await createRes.json();
    const taskId = created.task.id;

    // Now delete it
    const deleteRes = await invokeApp(app, {
      method: 'DELETE',
      path: `/api/scheduler/${taskId}`,
      headers: { 'X-API-Key': 'test-admin-key' },
    });
    expect(deleteRes.status).toBe(200);
    const deleteData = await deleteRes.json();
    expect(deleteData.success).toBe(true);
  });

  it('POST /api/scheduler/:id/toggle toggles task enabled state', async () => {
    // First create a task
    const createRes = await invokeApp(app, {
      method: 'POST',
      path: '/api/scheduler',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': 'test-admin-key',
      },
      body: {
        name: 'Task to Toggle',
        cron: '15 * * * *',
        command: 'echo "toggle me"',
      },
    });
    const created = await createRes.json();
    const taskId = created.task.id;
    const initialState = created.task.enabled;

    // Toggle it
    const toggleRes = await invokeApp(app, {
      method: 'POST',
      path: `/api/scheduler/${taskId}/toggle`,
      headers: { 'X-API-Key': 'test-admin-key' },
    });
    expect(toggleRes.status).toBe(200);
    const toggled = await toggleRes.json();
    expect(toggled.success).toBe(true);
    expect(toggled.enabled).toBe(!initialState);
  });
});
