import type { Request, Response } from 'express';
import { getEngine } from '../config.js';

export function activity(req: Request, res: Response): void {
  const limit = req.query.limit ? parseInt(req.query.limit as string, 10) : undefined;
  const entries = getEngine().getActivity(limit);
  res.json({ entries });
}
