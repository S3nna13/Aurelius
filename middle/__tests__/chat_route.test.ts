import { afterAll, afterEach, beforeAll, describe, expect, it, vi } from 'vitest';
import { request } from 'http';
import type { Server } from 'http';
import { buildApp } from '../src/server.js';

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

let server: Server;
const BASE = 'http://127.0.0.1:3097';

function sendJson(path: string, body: unknown) {
  return new Promise<{ status: number; text: string }>((resolve, reject) => {
    const payload = JSON.stringify(body);
    const req = request(
      `${BASE}${path}`,
      {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          'Content-Length': Buffer.byteLength(payload),
          'X-API-Key': 'test-admin-key',
        },
      },
      (res) => {
        const chunks: Buffer[] = [];
        res.on('data', (chunk) => chunks.push(Buffer.from(chunk)));
        res.on('end', () => {
          resolve({
            status: res.statusCode ?? 0,
            text: Buffer.concat(chunks).toString('utf8'),
          });
        });
      },
    );
    req.on('error', reject);
    req.write(payload);
    req.end();
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

  const app = buildApp();
  server = app.listen(3097, '127.0.0.1');
  await new Promise((resolve) => server.on('listening', resolve));
});

afterEach(() => {
  vi.clearAllMocks();
});

afterAll(() => {
  vi.unstubAllGlobals();
  server?.close();
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
    const payload = JSON.stringify({
      model: 'aurelius-1.3b',
      messages: [{ role: 'user', content: 'hello' }],
      stream: true,
    });

    const streamResult = await new Promise<{ status: number; text: string }>((resolve, reject) => {
      const req = request(
        `${BASE}/api/chat/completions`,
        {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
            'Content-Length': Buffer.byteLength(payload),
            'X-API-Key': 'test-admin-key',
          },
        },
        (res) => {
          const chunks: Buffer[] = [];
          res.on('data', (chunk) => chunks.push(Buffer.from(chunk)));
          res.on('end', () => {
            resolve({
              status: res.statusCode ?? 0,
              text: Buffer.concat(chunks).toString('utf8'),
            });
          });
        },
      );
      req.on('error', reject);
      req.write(payload);
      req.end();
    });

    expect(streamResult.status).toBe(200);
    expect(streamResult.text).toContain('data:');
    expect(streamResult.text).toContain('[DONE]');
  });
});
