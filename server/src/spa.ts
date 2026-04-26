import type { Request, Response, NextFunction } from 'express';
import path from 'path';
import fs from 'fs';
import { loadConfig } from './config.js';

const config = loadConfig();

export function spaFallback(req: Request, res: Response, next: NextFunction): void {
  if (req.path.startsWith('/api/') || req.path.startsWith('/assets/') || req.path.startsWith('/metrics')) {
    return next();
  }

  const indexHtml = path.join(config.frontendDist, 'index.html');
  if (fs.existsSync(indexHtml)) {
    res.sendFile(indexHtml);
  } else {
    res.status(200).type('html').send(`<!DOCTYPE html><html><head><title>Aurelius</title><style>body{font-family:system-ui;background:#0f0f1a;color:#e0e0e0;display:flex;align-items:center;justify-content:center;height:100vh;margin:0}p{font-size:1.2rem;opacity:0.6}</style></head><body><p>Aurelius Gateway — build frontend with <code>npm run build</code> in <code>frontend/</code></p></body></html>`);
  }
}
