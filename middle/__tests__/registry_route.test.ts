import { describe, expect, it } from 'vitest';
import { buildApp } from '../src/server.js';
import { loadRegistrySnapshot } from '../src/registry_bridge.js';
import { invokeApp } from './request-app.js';

const app = buildApp();
const snapshot = await loadRegistrySnapshot();

function requestJson(path: string) {
  return invokeApp(app, {
    method: 'GET',
    path,
    headers: {
      'X-API-Key': 'test-admin-key',
    },
  }).then(async (res) => ({
    status: res.status,
    body: res.text ? JSON.parse(res.text) : {},
  }));
}

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
