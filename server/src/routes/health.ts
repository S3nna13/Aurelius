import type { Request, Response } from 'express';
import type { SystemStats } from '../../../crates/data-engine/index.js';
import { getEngine } from '../config.js';

export function health(_req: Request, res: Response): void {
  const engine = getEngine();
  const fallbackStats: SystemStats = {
    agentCount: 0,
    activityCount: 0,
    notificationCount: 0,
    notificationUnread: 0,
    memoryEntryCount: 0,
    logCount: 0,
  };
  let stats: SystemStats = fallbackStats;
  try { stats = engine.getStats(); } catch { /* fallback */ }
  res.json({
    status: 'ok',
    time: Date.now() / 1000,
    version: '0.1.0',
    agents: stats.agentCount,
    activities: stats.activityCount,
    notifications: stats.notificationCount,
    unread: stats.notificationUnread,
    memory_entries: stats.memoryEntryCount,
    log_entries: stats.logCount,
  });
}
