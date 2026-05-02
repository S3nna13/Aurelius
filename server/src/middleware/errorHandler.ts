import type { Request, Response, NextFunction } from 'express';
import { logger } from './requestLog.js';

export function errorHandler(err: Error, _req: Request, res: Response, _next: NextFunction): void {
  logger.error({ err: { message: err.message, name: err.name, stack: err.stack } }, 'Unhandled error');
  res.status(500).json({ error: 'Internal server error' });
}
