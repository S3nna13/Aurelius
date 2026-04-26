import type { Request, Response, NextFunction } from 'express';
import { logger } from './requestLog.js';

export function errorHandler(err: Error, _req: Request, res: Response, _next: NextFunction): void {
  logger.error({ err }, 'Unhandled error');
  res.status(500).json({ error: 'Internal server error', message: err.message || 'Unknown error' });
}
