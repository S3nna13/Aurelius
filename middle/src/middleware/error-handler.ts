import type { Request, Response, NextFunction } from 'express'
import { getEngine } from '../engine.js'

export class AppError extends Error {
  constructor(
    public statusCode: number,
    message: string,
    public details?: unknown,
  ) {
    super(message)
    this.name = 'AppError'
  }
}

export function errorHandler(err: Error, _req: Request, res: Response, _next: NextFunction): void {
  const engine = getEngine()

  if (err instanceof AppError) {
    engine.appendLog('warn', 'http', `AppError ${err.statusCode}: ${err.message}`)
    res.status(err.statusCode).json({
      error: err.message,
      ...(err.details ? { details: err.details } : {}),
    })
    return
  }

  engine.appendLog('error', 'http', `Unhandled: ${err.message}`)
  console.error('[error]', err)

  res.status(500).json({
    error: 'Internal server error',
    requestId: _req.headers['x-request-id'] || undefined,
  })
}

export function notFoundHandler(_req: Request, res: Response): void {
  res.status(404).json({
    error: 'Not found',
    path: _req.originalUrl,
  })
}
