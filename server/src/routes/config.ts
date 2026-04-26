import type { Request, Response } from 'express';
import { getEngine } from '../config.js';

export function getConfig(_req: Request, res: Response): void {
  res.json({ config: getEngine().getAllConfig() });
}

export function setConfig(req: Request, res: Response): void {
  const { config } = req.body;
  if (!config || typeof config !== 'object') {
    res.status(400).json({ error: 'config must be an object' });
    return;
  }
  const engine = getEngine();
  for (const [key, value] of Object.entries(config)) {
    if (typeof value === 'string' || typeof value === 'number' || typeof value === 'boolean') {
      engine.setConfig(key, String(value));
    }
  }
  res.json({ success: true, config: engine.getAllConfig() });
}
