import { http, HttpResponse } from 'msw'

const BASE = '/api'

export const handlers = [
  http.get(`${BASE}/v1/models`, () =>
    HttpResponse.json({
      object: 'list',
      data: [{ id: 'aurelius', object: 'model', created: Date.now(), owned_by: 'aurelius' }],
    })
  ),

  http.get(`${BASE}/health`, () =>
    HttpResponse.json({ status: 'ok', uptime: 1234, version: '0.1.0', memory: { rss: 256 * 1024 * 1024 } })
  ),

  http.post(`${BASE}/chat/completions`, () =>
    HttpResponse.json({
      id: 'chatcmpl-test',
      object: 'chat.completion',
      created: Date.now(),
      model: 'aurelius',
      choices: [{ index: 0, message: { role: 'assistant', content: 'Test response' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 10, completion_tokens: 3, total_tokens: 13 },
    })
  ),

  http.post(`${BASE}/v1/chat/completions`, () =>
    HttpResponse.json({
      id: 'chatcmpl-test',
      object: 'chat.completion',
      created: Date.now(),
      model: 'aurelius',
      choices: [{ index: 0, message: { role: 'assistant', content: 'Test response' }, finish_reason: 'stop' }],
      usage: { prompt_tokens: 10, completion_tokens: 3, total_tokens: 13 },
    })
  ),
]
