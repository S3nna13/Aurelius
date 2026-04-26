import type { Request, Response } from 'express';
import { getEngine } from '../config.js';

export function listTrainingRuns(_req: Request, res: Response): void {
  const runs = getEngine().listTrainingRuns();
  const running = runs.filter((r: any) => r.status === 'running').length;
  const completed = runs.filter((r: any) => r.status === 'completed').length;
  res.json({ runs, summary: { total: runs.length, running, completed, queued: runs.filter((r: any) => r.status === 'queued').length } });
}

export function getTrainingRun(req: Request, res: Response): void {
  const run = getEngine().getTrainingRun(req.params.id as string);
  if (!run) {
    res.status(404).json({ error: `Training run ${req.params.id as string} not found` });
    return;
  }
  res.json(run);
}

export function createTrainingRun(req: Request, res: Response): void {
  const { name, model_id, total_epochs } = req.body;
  if (!name || !model_id) {
    res.status(400).json({ error: 'name and model_id are required' });
    return;
  }
  const engine = getEngine();
  const run = engine.createTrainingRun(name, model_id, total_epochs || 5);
  engine.appendLog('INFO', 'training', `Created training run: ${name}`);
  res.json({ success: true, run });
}
