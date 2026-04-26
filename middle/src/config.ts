import 'dotenv/config'

export const config = {
  host: process.env.MIDDLE_HOST || '0.0.0.0',
  port: parseInt(process.env.MIDDLE_PORT || '3001', 10),
  upstreamUrl: process.env.UPSTREAM_URL || 'http://127.0.0.1:8080',
  redisUrl: process.env.REDIS_URL || 'redis://localhost:6379/0',
  corsOrigin: process.env.CORS_ORIGIN || 'http://localhost:5173',
  jwtSecret: process.env.JWT_SECRET || 'aurelius-dev-secret',
  logLevel: process.env.MIDDLE_LOG_LEVEL || 'info',
  rateLimitRps: parseInt(process.env.RATE_LIMIT_RPS || '60', 10),
  rateLimitWindowMs: parseInt(process.env.RATE_LIMIT_WINDOW_MS || '60000', 10),
}
