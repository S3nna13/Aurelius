import type { Request, Response } from 'express';
import { getEngine } from '../config.js';
import { broadcastToChannel } from '../ws/hub.js';

export function listModels(_req: Request, res: Response): void {
  res.json({ models: getEngine().listModels() });
}

export function getModel(req: Request, res: Response): void {
  const model = getEngine().getModel(req.params.id as string);
  if (!model) {
    res.status(404).json({ error: `Model ${req.params.id as string} not found` });
    return;
  }
  res.json(model);
}

export function setModelState(req: Request, res: Response): void {
  const { state } = req.body;
  const id = req.params.id as string;
  if (!state || typeof state !== 'string') {
    res.status(400).json({ error: 'state is required' });
    return;
  }
  const engine = getEngine();
  const ok = engine.setModelState(id, state);
  if (!ok) {
    res.status(404).json({ error: `Model ${id} not found` });
    return;
  }
  engine.appendLog('INFO', 'models', `Model ${id} state -> ${state}`);
  broadcastToChannel('system', { type: 'model_state', model_id: id, state });
  res.json({ success: true, model: engine.getModel(id) });
}

export function getModelStats(_req: Request, res: Response): void {
  const models = getEngine().listModels();
  const loaded = models.filter((m: any) => m.state === 'loaded').length;
  const total = models.length;
  res.json({ loaded, total, models });
}
