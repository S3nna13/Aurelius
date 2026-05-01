import type { Request, Response } from 'express';
import { getEngine } from '../config.js';

export function memoryLayers(_req: Request, res: Response): void {
  const layers = getEngine().getMemoryLayers();
  const layerMap: Record<string, number> = {};
  const layerArr: { name: string; entries: number }[] = [];
  for (const l of layers) {
    layerMap[l.name] = l.entries;
    layerArr.push({ name: l.name, entries: l.entries });
  }
  res.json({ layers: layerMap, layerList: layerArr });
}

export function memoryEntries(req: Request, res: Response): void {
  const { layer, q, query, limit } = req.query;
  const searchQ = (q || query) as string | undefined;
  const entries = getEngine().getMemoryEntries(
    layer as string | undefined,
    searchQ,
    limit ? parseInt(limit as string, 10) : undefined,
  );
  res.json({ entries });
}

export function addMemoryEntry(req: Request, res: Response): void {
  const { layer, content } = req.body;
  if (!layer || !content) {
    res.status(400).json({ error: 'layer and content are required' });
    return;
  }
  const engine = getEngine();
  engine.addMemoryEntry(layer, content);
  engine.appendLog('INFO', 'memory', `Entry added to ${layer}`);
  res.json({ success: true, layer, content });
}
