import 'dotenv/config'

export type ChatBackend = 'mock' | 'vllm' | 'agentic'

const VALID_CHAT_BACKENDS: ChatBackend[] = ['mock', 'vllm', 'agentic']

export function normalizeChatBackend(
  value: string | null | undefined,
  fallback: ChatBackend = 'mock',
): ChatBackend {
  const normalized = value?.trim().toLowerCase()
  if (normalized && VALID_CHAT_BACKENDS.includes(normalized as ChatBackend)) {
    return normalized as ChatBackend
  }
  return fallback
}

export const config = {
  host: process.env.MIDDLE_HOST || '0.0.0.0',
  port: parseInt(process.env.MIDDLE_PORT || '3001', 10),
  upstreamUrl: process.env.UPSTREAM_URL || 'http://127.0.0.1:8080',
  vllmUpstreamUrl: process.env.AURELIUS_VLLM_URL || process.env.UPSTREAM_URL || 'http://127.0.0.1:8080',
  agenticUpstreamUrl:
    process.env.AURELIUS_AGENTIC_URL || process.env.UPSTREAM_URL || 'http://127.0.0.1:8080',
  defaultChatBackend: normalizeChatBackend(process.env.AURELIUS_DEFAULT_CHAT_BACKEND, 'mock'),
  redisUrl: process.env.REDIS_URL || 'redis://localhost:6379/0',
  corsOrigin: process.env.CORS_ORIGIN || 'http://localhost:5173',
  apiKey: process.env.AURELIUS_API_KEY || '',
  logLevel: process.env.MIDDLE_LOG_LEVEL || 'info',
  rateLimitRps: parseInt(process.env.RATE_LIMIT_RPS || '60', 10),
  rateLimitWindowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS || '60000', 10),
  allowPublicRegistration: process.env.ALLOW_PUBLIC_REGISTRATION === 'true' ? true : false,
  serviceApiKey: process.env.AURELIUS_SERVICE_KEY || process.env.AURELIUS_API_KEY || '',
}
