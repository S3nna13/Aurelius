import type { Request, Response, NextFunction } from 'express';
import crypto from 'crypto';
import type { GatewayConfig } from '../config.js';

export function authMiddleware(config: GatewayConfig) {
  return (req: Request, res: Response, next: NextFunction): void => {
    if (!config.authRequired) return next();
    if (req.path === '/api/health' || req.path === '/api/license/validate') return next();

    const apiKey = req.headers['x-api-key'] as string | undefined;
    if (!apiKey) {
      res.status(401).json({ error: 'Unauthorized', message: 'X-API-Key header required' });
      return;
    }

    if (!config.apiKey) {
      res.status(500).json({ error: 'Server misconfigured', message: 'API key not configured' });
      return;
    }

    const bufKey = Buffer.from(apiKey);
    const bufExpected = Buffer.from(config.apiKey);
    if (bufKey.length !== bufExpected.length || !crypto.timingSafeEqual(bufKey, bufExpected)) {
      res.status(401).json({ error: 'Unauthorized', message: 'Invalid API key' });
      return;
    }

    next();
  };
}
