import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { buildApp } from '../src/server.js';
import { invokeApp } from './request-app.js';

const mockCompletion = {
  id: 'chatcmpl-test',
  object: 'chat.completion',
  created: 1714300000,
  model: 'aurelius-1.3b',
  choices: [
    {
      index: 0,
      message: { role: 'assistant', content: 'Live Aurelius answer' },
      finish_reason: 'stop',
    },
  ],
};

let app: ReturnType<typeof buildApp>;

function sendJson(path: string, body: unknown) {
  return invokeApp(app, {
    method: 'POST',
    path,
    headers: {
      'Content-Type': 'application/json',
      'X-API-Key': 'test-admin-key',
    },
    body,
  });
}

beforeAll(async () => {
  vi.stubGlobal(
    'fetch',
    vi.fn(async () => new Response(JSON.stringify(mockCompletion), {
      status: 200,
      headers: { 'Content-Type': 'application/json' },
    })),
  );

  app = buildApp();
});

afterEach(() => {
  vi.clearAllMocks();
});

afterAll(() => {
  vi.unstubAllGlobals();
});

describe('chat route', () => {
  it('returns a live upstream completion', async () => {
    const res = await sendJson('/api/chat/completions', {
      model: 'aurelius-1.3b',
      messages: [{ role: 'user', content: 'hello' }],
    });

    expect(res.status).toBe(200);
    const data = JSON.parse(res.text);
    expect(data.choices[0].message.content).toContain('Live Aurelius answer');
  });

  it('streams upstream completion content as SSE', async () => {
    const streamResult = await invokeApp(app, {
      method: 'POST',
      path: '/api/chat/completions',
      headers: {
        'Content-Type': 'application/json',
        'X-API-Key': 'test-admin-key',
      },
      body: {
        model: 'aurelius-1.3b',
        messages: [{ role: 'user', content: 'hello' }],
        stream: true,
      },
    });

    expect(streamResult.status).toBe(200);
    expect(streamResult.text).toContain('data:');
    expect(streamResult.text).toContain('[DONE]');
  });
});
