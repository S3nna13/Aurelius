import type { Request, Response } from 'express';
import { getEngine } from '../config.js';

export function memoryLayers(_req: Request, res: Response): void {
  const layers = getEngine().getMemoryLayers();
  const layerMap: Record<string, number> = {};
  for (const l of layers) layerMap[l.name] = l.entries;
  res.json({ layers: layerMap });
}

export function memoryEntries(req: Request, res: Response): void {
  const { layer, q, limit } = req.query;
  const entries = getEngine().getMemoryEntries(
    layer as string | undefined,
    q as string | undefined,
    limit ? parseInt(limit as string, 10) : undefined,
  );
  res.json({ entries });
}
