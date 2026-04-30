import crypto from 'crypto'
import type { Request, Response, NextFunction } from 'express'

export interface AuthUser {
  id: string
  role: 'admin' | 'user' | 'agent'
  scopes: string[]
}

declare global {
  namespace Express {
    interface Request {
      user?: AuthUser
    }
  }
}

const API_KEYS = new Map<string, AuthUser>()

const DEFAULT_KEY = process.env.AURELIUS_API_KEY || 'dev-key'
API_KEYS.set(DEFAULT_KEY, { id: 'admin', role: 'admin', scopes: ['*'] })

function timingSafeCompare(a: string, b: string): boolean {
  if (a.length !== b.length) return false
  try {
    return crypto.timingSafeEqual(Buffer.from(a), Buffer.from(b))
  } catch {
    return false
  }
}

export function registerApiKey(key: string, user: AuthUser): void {
  API_KEYS.set(key, user)
}

export function authMiddleware(req: Request, res: Response, next: NextFunction): void {
  const publicPaths = ['/health', '/healthz', '/readyz', '/openapi.json', '/docs']

  if (publicPaths.includes(req.path)) {
    next()
    return
  }

  const authHeader = req.headers['x-api-key'] || req.headers['authorization']
  const apiKey = typeof authHeader === 'string'
    ? authHeader.replace(/^Bearer\s+/i, '').trim()
    : null

  if (apiKey) {
    for (const [key, user] of API_KEYS) {
      if (timingSafeCompare(apiKey, key)) {
        req.user = user
        next()
        return
      }
    }
  }

  res.status(401).json({ error: 'Unauthorized', message: 'Valid API key required' })
}

export function requireScope(...scopes: string[]) {
  return (req: Request, res: Response, next: NextFunction): void => {
    if (!req.user) {
      res.status(401).json({ error: 'Unauthorized' })
      return
    }
    if (req.user.scopes.includes('*') || scopes.some((s) => req.user!.scopes.includes(s))) {
      next()
      return
    }
    res.status(403).json({ error: 'Forbidden', message: 'Insufficient permissions' })
  }
}
