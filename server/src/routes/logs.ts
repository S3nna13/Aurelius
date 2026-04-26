import type { Request, Response } from 'express';
import { getEngine } from '../config.js';

export function getLogs(req: Request, res: Response): void {
  const { level, q, limit } = req.query;
  const entries = getEngine().getLogs(
    level as string | undefined,
    q as string | undefined,
    limit ? parseInt(limit as string, 10) : undefined,
  );
  res.json({ entries });
}
