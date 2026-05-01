import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { request } from 'http';
import type { Server } from 'http';
import { buildApp } from '../src/server.js';
import { loadRegistrySnapshot } from '../src/registry_bridge.js';

let server: Server;
const BASE = 'http://127.0.0.1:3096';
const snapshot = await loadRegistrySnapshot();

function requestJson(path: string) {
  return new Promise<{ status: number; body: any }>((resolve, reject) => {
    const req = request(
      `${BASE}${path}`,
      {
        method: 'GET',
        headers: {
          'X-API-Key': 'test-admin-key',
        },
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on('data', (chunk) => chunks.push(Buffer.from(chunk)));
        res.on('end', () => {
          const text = Buffer.concat(chunks).toString('utf8');
          resolve({
            status: res.statusCode ?? 0,
            body: text ? JSON.parse(text) : {},
          });
        });
      },
    );
    req.on('error', reject);
    req.end();
  });
}

beforeAll(async () => {
  const app = buildApp();
  server = app.listen(3096, '127.0.0.1');
  await new Promise((resolve) => server.on('listening', resolve));
});

afterAll(() => {
  server?.close();
});

describe('registry routes', () => {
  it('serves the live agent registry', async () => {
    const res = await requestJson('/api/registry/agents');

    expect(res.status).toBe(200);
    expect(res.body.total).toBe(snapshot.agents.length);
    expect(res.body.categories).toEqual(snapshot.agent_categories);
    expect(res.body.agents.find((agent: any) => agent.id === 'coding')).toBeDefined();
  });

  it('filters live agents by category', async () => {
    const res = await requestJson('/api/registry/agents?category=coding');

    expect(res.status).toBe(200);
    expect(res.body.agents.every((agent: any) => agent.category === 'coding')).toBe(true);
  });

  it('serves live skill details', async () => {
    const res = await requestJson('/api/registry/skills/code_review');

    expect(res.status).toBe(200);
    expect(res.body.skill.id).toBe('code_review');
    expect(res.body.skill.name).toBe('Code Review');
  });

  it('serves the live skills registry', async () => {
    const res = await requestJson('/api/registry/skills');

    expect(res.status).toBe(200);
    expect(res.body.total).toBe(snapshot.skills.length);
    expect(res.body.categories).toEqual(snapshot.skill_categories);
    expect(res.body.skills.find((skill: any) => skill.id === 'workflow_automation')).toBeDefined();
  });
});
