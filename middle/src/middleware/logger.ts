import type { Request, Response, NextFunction } from 'express'
import { getEngine } from '../engine.js'

export function requestLogger(req: Request, res: Response, next: NextFunction): void {
  const start = Date.now()
  res.on('finish', () => {
    const duration = Date.now() - start
    const msg = `${req.method} ${req.originalUrl} ${res.statusCode} ${duration}ms`
    const level = res.statusCode >= 500 ? 'error' : res.statusCode >= 400 ? 'warn' : 'info'
    getEngine().appendLog(level, 'http', msg)
  })
  next()
}
