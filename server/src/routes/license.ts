import type { Request, Response } from 'express';
import { getEngine } from '../config.js';

export function validateLicense(_req: Request, res: Response): void {
  const engine = getEngine();
  const activated = engine.getConfig('license_activated') === 'true';
  const key = engine.getConfig('license_key') || '';
  const tier = engine.getConfig('license_tier') || 'trial';
  res.json({ valid: activated && !!key, activated, tier });
}

export function activateLicense(req: Request, res: Response): void {
  const { license_key, tier } = req.body;
  if (!license_key) {
    res.status(400).json({ error: 'license_key required' });
    return;
  }
  if (license_key.startsWith('AURELIUS-') && license_key.length >= 32) {
    const engine = getEngine();
    engine.setConfig('license_key', license_key);
    engine.setConfig('license_activated', 'true');
    engine.setConfig('license_tier', tier || 'pro');
    engine.setConfig('require_auth', 'true');
    engine.setConfig('api_key', license_key.slice(-16));
    res.json({ success: true, tier: tier || 'pro' });
  } else {
    res.status(403).json({ error: 'Invalid license key' });
  }
}
