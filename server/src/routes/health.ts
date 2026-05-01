import type { Request, Response } from 'express';
import { getEngine } from '../config.js';

export function health(_req: Request, res: Response): void {
  const engine = getEngine();
  let stats = { agent_count: 0, activity_count: 0, notification_count: 0, notification_unread: 0, memory_entry_count: 0, log_count: 0 };
  try { stats = engine.getStats(); } catch { /* fallback */ }
  res.json({
    status: 'ok',
    time: Date.now() / 1000,
    version: '0.1.0',
    agents: stats.agent_count,
    activities: stats.activity_count,
    notifications: stats.notification_count,
    unread: stats.notification_unread,
    memory_entries: stats.memory_entry_count,
    log_entries: stats.log_count,
  });
}
