import type { Request, Response, NextFunction } from 'express'
import { config } from '../config.js'

interface Bucket {
  tokens: number
  lastRefill: number
}

const buckets = new Map<string, Bucket>()

export function rateLimiter(req: Request, res: Response, next: NextFunction): void {
  const key = req.user?.id || req.ip || 'anonymous'
  const now = Date.now()
  let bucket = buckets.get(key)

  if (!bucket) {
    bucket = { tokens: config.rateLimitRps, lastRefill: now }
    buckets.set(key, bucket)
  }

  const elapsed = now - bucket.lastRefill
  const refill = Math.floor(elapsed / config.rateLimitWindowMs) * config.rateLimitRps
  if (refill > 0) {
    bucket.tokens = Math.min(config.rateLimitRps, bucket.tokens + refill)
    bucket.lastRefill = now
  }

  if (bucket.tokens <= 0) {
    res.status(429).json({
      error: 'Rate limit exceeded',
      retryAfter: Math.ceil((config.rateLimitWindowMs - elapsed) / 1000),
    })
    return
  }

  bucket.tokens -= 1
  res.setHeader('X-RateLimit-Remaining', bucket.tokens)
  next()
}

setInterval(() => {
  const now = Date.now()
  for (const [key, bucket] of buckets) {
    if (now - bucket.lastRefill > config.rateLimitWindowMs * 2) {
      buckets.delete(key)
    }
  }
}, 60000).unref()
