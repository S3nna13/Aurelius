import type { Request, Response } from 'express';
import { getEngine } from '../config.js';

const PROTECTED_CONFIG_KEYS = new Set(['api_key', 'license_key', 'license_activated', 'require_auth']);

export function getConfig(_req: Request, res: Response): void {
  const raw = getEngine().getAllConfig();
  const config = Object.fromEntries(
    Object.entries(raw).filter(([k]) => !PROTECTED_CONFIG_KEYS.has(k))
  );
  res.json({ config });
}

export function setConfig(req: Request, res: Response): void {
  const { config } = req.body;
  if (!config || typeof config !== 'object') {
    res.status(400).json({ error: 'config must be an object' });
    return;
  }
  const engine = getEngine();
  for (const [key, value] of Object.entries(config)) {
    if (PROTECTED_CONFIG_KEYS.has(key)) {
      res.status(403).json({ error: `Config key '${key}' is protected and cannot be set via API` });
      return;
    }
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
      engine.setConfig(key, String(value));
    }
  }
  const raw = engine.getAllConfig();
  const safeConfig = Object.fromEntries(
    Object.entries(raw).filter(([k]) => !PROTECTED_CONFIG_KEYS.has(k))
  );
  res.json({ success: true, config: safeConfig });
}
