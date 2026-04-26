import type { Request, Response } from 'express';
import { getEngine } from '../config.js';
import { broadcastToChannel } from '../ws/hub.js';

export function command(req: Request, res: Response): void {
  const { command: text } = req.body;
  if (typeof text !== 'string') {
    res.status(400).json({ error: 'command must be a string' });
    return;
  }

  const engine = getEngine();

  try {
    const action = 'echo';
    const target = text;
    const output = `Echo: ${text}`;

    engine.appendActivity(text, true, output);
    engine.appendLog('INFO', 'gateway', `Command executed: ${text}`);
    broadcastToChannel('system', { type: 'activity', command: text, success: true });

    res.json({ success: true, output, action, target });
  } catch (err: unknown) {
    const msg = err instanceof Error ? err.message : String(err);
    engine.appendActivity(text, false, msg);
    engine.appendLog('ERROR', 'gateway', `Command failed: ${msg}`);

    res.status(500).json({ success: false, error: msg });
  }
}
