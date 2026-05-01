import type { Request, Response } from 'express';
import { getEngine } from '../config.js';
import { broadcastToChannel } from '../ws/hub.js';

export function status(_req: Request, res: Response): void {
  const engine = getEngine();
  const agents = engine.listAgents();
  const memoryLayers = engine.getMemoryLayers();
  const notifStats = engine.getNotificationStats();

  const agentsOnline = agents.filter(a =>
    ['ACTIVE', 'RUNNING', 'IDLE'].includes(a.state.toUpperCase())
  ).length;

  res.json({
    agents,
    skills: [
      { id: 'code-review', active: true },
      { id: 'refactor', active: true },
      { id: 'test-gen', active: false },
    ],
    plugins: [{ id: 'mcp-core' }],
    memory: { total_entries: memoryLayers.reduce((s, l) => s + l.entries, 0) },
    notifications: { unread: notifStats.unread },
    counts: {
      agents_online: agentsOnline,
      agents_total: agents.length,
      skills_active: 2,
      skills_total: 3,
      plugins_total: 1,
      notifications_unread: notifStats.unread,
    },
  });
}

export function agentDetail(req: Request, res: Response): void {
  const engine = getEngine();
  const agentId = req.params.id as string;
  const agent = engine.getAgent(agentId);
  if (!agent) {
    res.status(404).json({ error: 'Agent not found' });
    return;
  }
  res.json(agent);
}

export function updateAgent(req: Request, res: Response): void {
  const engine = getEngine();
  const agentId = req.params.id as string;
  const { state, role, metrics_json } = req.body;

  const existing = engine.getAgent(agentId);
  if (!existing) {
    res.status(404).json({ error: 'Agent not found' });
    return;
  }

  engine.upsertAgent(
    agentId,
    state || existing.state,
    role || existing.role,
    metrics_json || existing.metricsJson,
  );

  const updated = engine.getAgent(agentId);
  broadcastToChannel('agents', { type: 'change', agent: updated });

  engine.appendLog('INFO', 'gateway', `Agent ${agentId} state → ${state || existing.state}`);
  res.json(updated);
}
