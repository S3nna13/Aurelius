import type { Request, Response, NextFunction } from 'express'
import { getEngine } from '../engine.js'

const SENSITIVE_PARAMS = ['api_key', 'apiKey', 'token', 'password', 'secret', 'key', 'auth', 'authorization', 'x-api-key']

function redactQuery(url: string): string {
  try {
    const u = new URL(url, 'http://localhost')
    for (const param of SENSITIVE_PARAMS) {
      if (u.searchParams.has(param)) {
        u.searchParams.set(param, '[REDACTED]')
      }
    }
    return u.pathname + u.search
  } catch {
    return url
  }
}

export function requestLogger(req: Request, res: Response, next: NextFunction): void {
  const start = Date.now()
  res.on('finish', () => {
    const duration = Date.now() - start
    const safeUrl = redactQuery(req.originalUrl)
    const msg = `${req.method} ${safeUrl} ${res.statusCode} ${duration}ms`
    const level = res.statusCode >= 500 ? 'error' : res.statusCode >= 400 ? 'warn' : 'info'
    getEngine().appendLog(level, 'http', msg)
  })
  next()
}
