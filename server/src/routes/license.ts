import type { Request, Response } from 'express';
import crypto from 'crypto';
import { getEngine } from '../config.js';

// Basic license key format: AURELIUS- followed by hex characters (32+ total length after prefix)
const LICENSE_KEY_RE = /^AURELIUS-(?:[A-Fa-f0-9]{4,}){4,}$/;

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
  // Harden format check beyond just startsWith + length
  if (!LICENSE_KEY_RE.test(license_key)) {
    res.status(403).json({ error: 'Invalid license key format' });
    return;
  }
  const engine = getEngine();
  engine.setConfig('license_key', license_key);
  engine.setConfig('license_activated', 'true');
  engine.setConfig('license_tier', tier || 'pro');
  engine.setConfig('require_auth', 'true');
  // Derive API key using HMAC rather than raw slice
  const hmac = crypto.createHmac('sha256', license_key).update('aurelius-api-key').digest('hex');
  engine.setConfig('api_key', hmac.slice(0, 32));
  res.json({ success: true, tier: tier || 'pro' });
}
