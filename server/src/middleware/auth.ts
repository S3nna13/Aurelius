import type { Request, Response, NextFunction } from 'express';
import type { GatewayConfig } from '../config.js';

export function validateAuthConfig(config: GatewayConfig): void {
  if (config.authRequired && !config.apiKey) {
    throw new Error('AURELIUS_API_KEY must be set when auth is required. Set AURELIUS_DEV_OPEN=true to disable auth in development.');
  }
}

export function authMiddleware(config: GatewayConfig) {
  return (req: Request, res: Response, next: NextFunction): void => {
    if (!config.authRequired) return next();
    if (req.path === '/api/health' || req.path === '/api/license/validate') return next();

    const apiKey = req.headers['x-api-key'] as string | undefined;
    if (!apiKey) {
      res.status(401).json({ error: 'Unauthorized', message: 'X-API-Key header required' });
      return;
    }

    if (config.apiKey && apiKey !== config.apiKey) {
      res.status(401).json({ error: 'Unauthorized', message: 'Invalid API key' });
      return;
    }

    next();
  };
}
