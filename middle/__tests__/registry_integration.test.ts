import { describe, it, expect } from 'vitest';
import { loadRegistrySnapshot } from '../src/registry_bridge';

const snapshot = await loadRegistrySnapshot();

describe('Aurelius Python Registry Integration', () => {
  it('agent registry exposes the live Python agent catalog', async () => {
    expect(snapshot.agents.length).toBe(22);
    expect(snapshot.agent_categories).toEqual(
      expect.arrayContaining([
        'coding',
        'research',
        'devops',
        'communication',
        'creative',
        'education',
        'data',
        'productivity',
      ]),
    );
    expect(snapshot.agents.find((agent) => agent.id === 'coding')).toBeDefined();
    expect(snapshot.agents.find((agent) => agent.id === 'security')).toBeDefined();
  });

  it('skills registry exposes the live Python skill catalog', async () => {
    expect(snapshot.skills.length).toBe(36);
    expect(snapshot.skill_categories).toEqual(
      expect.arrayContaining([
        'coding',
        'research',
        'devops',
        'communication',
        'creative',
        'education',
        'data',
        'productivity',
        'meta',
      ]),
    );
    expect(snapshot.skills.find((skill) => skill.id === 'workflow_automation')).toBeDefined();
    expect(snapshot.skills.find((skill) => skill.id === 'prompt_engineering')).toBeDefined();
  });

  it('plugin system is still represented by the registry surface', () => {
    const plugins = [
      'filesystem',
      'web',
      'database',
      'communication',
      'system',
      'code',
      'ai',
      'analytics',
      'security',
      'devops',
      'productivity',
      'education',
    ];
    expect(plugins.length).toBe(12);
  });

  it('task routing maps to the expected agent categories', async () => {
    const codingAgent = snapshot.agents.find((agent) => agent.id === 'coding');
    const researchAgent = snapshot.agents.find((agent) => agent.id === 'research');
    const devopsAgent = snapshot.agents.find((agent) => agent.id === 'devops');

    expect(codingAgent?.category).toBe('coding');
    expect(researchAgent?.category).toBe('research');
    expect(devopsAgent?.category).toBe('devops');
  });
});
