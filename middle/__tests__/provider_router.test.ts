import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { getEngine } from '../src/engine';
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

let fetchMock: ReturnType<typeof vi.fn>;

describe('ProviderRouter', () => {
  beforeEach(() => {
    fetchMock = vi.fn(async () => new Response(JSON.stringify(mockCompletion), {
        status: 200,
        headers: { 'Content-Type': 'application/json' },
      }));
    vi.stubGlobal('fetch', fetchMock);
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

  it('routes mock backend without calling upstream', async () => {
    const router = new ProviderRouter();
    const result = await router.complete({
      model: 'aurelius-1.3b',
      backend: 'mock',
      messages: [{ role: 'user', content: 'hello mock backend' }],
    });

    expect(result.choices[0].message.content).toContain('Mock response to: hello mock backend');
    expect(fetchMock).not.toHaveBeenCalled();
    expect(router.getStats().mock.requests).toBe(1);
  });

  it('routes vllm backend to the configured upstream url', async () => {
    const router = new ProviderRouter({ vllmUpstreamUrl: 'http://vllm-upstream.local' });
    await router.complete({
      model: 'aurelius-1.3b',
      backend: 'vllm',
      messages: [{ role: 'user', content: 'hello vllm' }],
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe('http://vllm-upstream.local/v1/chat/completions');
  });

  it('routes agentic backend to the configured upstream url', async () => {
    const router = new ProviderRouter({ agenticUpstreamUrl: 'http://agentic-upstream.local' });
    await router.complete({
      model: 'aurelius-1.3b',
      backend: 'agentic',
      messages: [{ role: 'user', content: 'hello agentic' }],
    });

    expect(fetchMock).toHaveBeenCalledTimes(1);
    expect(fetchMock.mock.calls[0][0]).toBe('http://agentic-upstream.local/v1/chat/completions');
  });

  it('uses the configured default backend when none is specified', async () => {
    const engine = getEngine();
    const previousBackend = engine.getConfig('chat.backend');
    const previousModel = engine.getConfig('chat.model');
    const previousTemperature = engine.getConfig('chat.temperature');
    engine.setConfig('chat.backend', 'agentic');
    engine.setConfig('chat.model', 'aurelius-2.7b');
    engine.setConfig('chat.temperature', '0.42');

    try {
      const router = new ProviderRouter({ agenticUpstreamUrl: 'http://agentic-upstream.local' });
      await router.complete({
        messages: [{ role: 'user', content: 'hello default backend' }],
      });

      expect(fetchMock).toHaveBeenCalledTimes(1);
      expect(fetchMock.mock.calls[0][0]).toBe('http://agentic-upstream.local/v1/chat/completions');
      const init = fetchMock.mock.calls[0][1] as RequestInit | undefined;
      const payload = JSON.parse(String(init?.body ?? '{}')) as Record<string, unknown>;
      expect(payload.model).toBe('aurelius-2.7b');
      expect(payload.temperature).toBe(0.42);
    } finally {
      engine.setConfig('chat.backend', previousBackend || 'vllm');
      engine.setConfig('chat.model', previousModel || 'aurelius-1.3b');
      engine.setConfig('chat.temperature', previousTemperature || '0.7');
    }
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

  it('resolveBackend with explicit auto falls back to config default', () => {
    const engine = getEngine();
    const previousBackend = engine.getConfig('chat.backend');
    engine.setConfig('chat.backend', 'agentic');

    try {
      const router = new ProviderRouter();
      const result = router.resolveBackend('auto');
      expect(result.backend).toBe('agentic');
      expect(result.resolved).toBe('agentic');
    } finally {
      engine.setConfig('chat.backend', previousBackend || 'vllm');
    }
  });

  it('resolveBackend with invalid backend value returns config default (normalization)', () => {
    const engine = getEngine();
    const previousBackend = engine.getConfig('chat.backend');
    engine.setConfig('chat.backend', 'vllm');

    try {
      const router = new ProviderRouter();
      const result = router.resolveBackend('invalid-backend');
      expect(result.backend).toBe('vllm');
      expect(result.resolved).toBe('vllm');
    } finally {
      engine.setConfig('chat.backend', previousBackend || 'vllm');
    }
  });

  it('complete returns resolved_backend in response when using auto', async () => {
    const engine = getEngine();
    const previousBackend = engine.getConfig('chat.backend');
    engine.setConfig('chat.backend', 'vllm');

    try {
      const router = new ProviderRouter({ vllmUpstreamUrl: 'http://vllm-upstream.local' });
      const result = await router.complete({
        model: 'aurelius-1.3b',
        backend: 'auto',
        messages: [{ role: 'user', content: 'hello auto backend' }],
      });

      expect(result.resolved_backend).toBe('vllm');
    } finally {
      engine.setConfig('chat.backend', previousBackend || 'vllm');
    }
  });

  it('complete returns resolved_backend with explicit backend value', async () => {
    const router = new ProviderRouter({ vllmUpstreamUrl: 'http://vllm-upstream.local' });
    const result = await router.complete({
      model: 'aurelius-1.3b',
      backend: 'vllm',
      messages: [{ role: 'user', content: 'hello explicit vllm' }],
    });

    expect(result.resolved_backend).toBe('vllm');
  });
});
