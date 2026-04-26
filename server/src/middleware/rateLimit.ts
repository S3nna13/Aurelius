import rateLimit from 'express-rate-limit';
import type { GatewayConfig } from '../config.js';

export function rateLimitMiddleware(config: GatewayConfig) {
  return rateLimit({
    windowMs: config.rateLimitWindowMs,
    max: config.rateLimitMax,
    standardHeaders: true,
    legacyHeaders: false,
    message: { error: 'Rate limit exceeded', message: 'Too many requests, try again later' },
  });
}
