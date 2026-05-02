import { createRequire } from 'module';
import path from 'path';
import { fileURLToPath } from 'url';
import { DataEngine } from '../../crates/data-engine/index.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export interface GatewayConfig {
  host: string;
  port: number;
  pythonUrl: string;
  authRequired: boolean;
  apiKey: string;
  frontendDist: string;
  rateLimitWindowMs: number;
  rateLimitMax: number;
}

export function loadConfig(): GatewayConfig {
  return {
    host: process.env.AURELIUS_HOST || '0.0.0.0',
    port: parseInt(process.env.AURELIUS_PORT || '7870', 10),
    pythonUrl: process.env.AURELIUS_PYTHON_URL || 'http://127.0.0.1:8080',
    authRequired: process.env.AURELIUS_AUTH === 'true' || (process.env.NODE_ENV === 'production' && process.env.AURELIUS_DEV_OPEN !== 'true'),
    apiKey: process.env.AURELIUS_API_KEY || '',
    frontendDist: process.env.AURELIUS_FRONTEND_DIST
      || path.resolve(__dirname, '../../frontend/dist'),
    rateLimitWindowMs: 60_000,
    rateLimitMax: 200,
  };
}

const _engine = new DataEngine();
export function getEngine(): DataEngine {
  return _engine;
}
