import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { ProviderRouter } from '../src/provider_router';

const mockCompletion = {
  id: 'chatcmpl-test',
  object: 'chat.completion',
  created: 1714300000,
  model: 'aurelius-1.3b',
  choices: [
    {
      index: 0,
      message: { role: 'assistant', content: 'Aurelius live reply' },
      finish_reason: 'stop',
    },
  ],
  usage: {
    prompt_tokens: 8,
    completion_tokens: 4,
    total_tokens: 12,
  },
};

describe('ProviderRouter', () => {
  beforeEach(() => {
    vi.stubGlobal(
      'fetch',
      vi.fn(async () => new Response(JSON.stringify(mockCompletion), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      })),
    );
  });

  afterEach(() => {
    vi.unstubAllGlobals();
    vi.restoreAllMocks();
  });

  it('routes to Aurelius first', async () => {
    const router = new ProviderRouter();
    const result = await router.complete({
      model: 'aurelius-1.3b',
      messages: [{ role: 'user', content: 'hello' }],
    });
    expect(result.choices[0].message.content).toContain('Aurelius live reply');
  });

  it('returns available models across providers', () => {
    const router = new ProviderRouter();
    const models = router.getAvailableModels();
    expect(models.length).toBeGreaterThan(0);
    const providers = [...new Set(models.map((m) => m.provider))];
    expect(providers).toContain('aurelius');
  });

  it('tracks stats per provider', async () => {
    const router = new ProviderRouter();
    await router.complete({
      model: 'aurelius-1.3b',
      messages: [{ role: 'user', content: 'hi' }],
    });
    const stats = router.getStats();
    expect(stats.aurelius.requests).toBe(1);
  });

  it('supports chat completion interface', async () => {
    const router = new ProviderRouter();
    const result = await router.chat([{ role: 'user', content: 'hello' }], { model: 'aurelius-1.3b' });
    expect(result.choices).toBeDefined();
  });

  it('loads and unloads models', () => {
    const router = new ProviderRouter();
    expect(router.loadModel('test-model')).toBe(true);
    expect(router.unloadModel('test-model')).toBe(true);
  });
});
