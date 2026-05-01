import type { Request, Response } from 'express';
import { getEngine } from '../config.js';
import { broadcastToChannel } from '../ws/hub.js';

export function listTrainingRuns(_req: Request, res: Response): void {
  const runs = getEngine().listTrainingRuns();
  const running = runs.filter((r: any) => r.status === 'running').length;
  const completed = runs.filter((r: any) => r.status === 'completed').length;
  res.json({ runs, summary: { total: runs.length, running, completed, queued: runs.filter((r: any) => r.status === 'queued').length } });
}

export function trainingStats(_req: Request, res: Response): void {
  const runs = getEngine().listTrainingRuns();
  const totalSteps = runs.reduce((s: number, r: any) => s + r.totalSteps, 0);
  const totalEpochs = runs.reduce((s: number, r: any) => s + r.currentEpoch, 0);
  const avgValLoss = runs.filter((r: any) => r.bestValLoss > 0).reduce((s: number, r: any) => s + r.bestValLoss, 0) /
    Math.max(1, runs.filter((r: any) => r.bestValLoss > 0).length);
  const bestLoss = runs.filter((r: any) => r.bestValLoss > 0).reduce(
    (m: number, r: any) => Math.min(m, r.bestValLoss), Infinity);
  const running = runs.filter((r: any) => r.status === 'running');
  const activeRuns = running.map((r: any) => ({ id: r.id, name: r.name, currentEpoch: r.currentEpoch, totalEpochs: r.totalEpochs, currentLr: r.currentLr }));

  res.json({
    total_runs: runs.length,
    total_steps: totalSteps,
    total_epochs: totalEpochs,
    avg_val_loss: avgValLoss > 0 ? parseFloat(avgValLoss.toFixed(4)) : null,
    best_val_loss: bestLoss < Infinity ? parseFloat(bestLoss.toFixed(4)) : null,
    running_count: running.length,
    active_runs: activeRuns,
  });
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
  broadcastToChannel('training', { type: 'created', run });
  res.json({ success: true, run });
}
