import crypto from 'crypto'
import type { Request, Response, NextFunction } from 'express'
import { config } from '../config.js'

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

if (!config.apiKey) {
  throw new Error('AURELIUS_API_KEY environment variable is required')
}
API_KEYS.set(config.apiKey, { id: 'admin', role: 'admin', scopes: ['*'] })

function timingSafeCompare(a: string, b: string): boolean {
  const bufA = Buffer.from(a)
  const bufB = Buffer.from(b)
  if (bufA.length !== bufB.length) {
    const longer = bufA.length > bufB.length ? bufA : bufB
    const shorter = bufA.length > bufB.length ? bufB : bufA
    crypto.timingSafeEqual(longer.subarray(0, shorter.length), shorter)
    return false
  }
  return crypto.timingSafeEqual(bufA, bufB)
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

  const apiKeyHeader = req.headers['x-api-key']
  const apiKey = typeof apiKeyHeader === 'string' ? apiKeyHeader.trim() : null

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
