import { Router, Request, Response } from 'express';
import { v4 as uuid } from 'uuid';

const router = Router();

// In-memory agent store
interface AgentRecord {
  id: string;
  name: string;
  role: string;
  capabilities: string[];
  state: 'idle' | 'active' | 'busy' | 'error' | 'terminated';
  created: number;
  lastHeartbeat: number;
  metrics: { tasksCompleted: number; avgLatencyMs: number; errorRate: number };
}

const agents = new Map<string, AgentRecord>();

// SSE connections per agent
const agentStreams = new Map<string, Set<(event: string, data: unknown) => void>>();

function broadcast(agentId: string, event: string, data: unknown) {
  const subs = agentStreams.get(agentId);
  if (subs) subs.forEach(fn => fn(event, data));
}

// GET /api/agents
router.get('/', (_req: Request, res: Response) => {
  res.json({ agents: Array.from(agents.values()) });
});

// GET /api/agents/:id
router.get('/:id', (req: Request, res: Response) => {
  const agentId = String(req.params.id);
  const agent = agents.get(agentId);
  if (!agent) return res.status(404).json({ error: 'Agent not found' });
  res.json({ agent });
});

// POST /api/agents
router.post('/', (req: Request, res: Response) => {
  const { name, role, capabilities } = req.body;
  if (!name) return res.status(400).json({ error: 'name required' });

  const agent: AgentRecord = {
    id: uuid().slice(0, 12),
    name,
    role: role || 'worker',
    capabilities: capabilities || [],
    state: 'idle',
    created: Date.now(),
    lastHeartbeat: Date.now(),
    metrics: { tasksCompleted: 0, avgLatencyMs: 0, errorRate: 0 },
  };
  agents.set(agent.id, agent);
  broadcast(agent.id, 'created', agent);
  res.status(201).json({ agent });
});

// POST /api/agents/:id/heartbeat
router.post('/:id/heartbeat', (req: Request, res: Response) => {
  const agentId = String(req.params.id);
  const agent = agents.get(agentId);
  if (!agent) return res.status(404).json({ error: 'Agent not found' });
  agent.lastHeartbeat = Date.now();
  agent.state = req.body.state || agent.state;
  broadcast(agent.id, 'heartbeat', agent);
  res.json({ ok: true });
});

// DELETE /api/agents/:id
router.delete('/:id', (req: Request, res: Response) => {
  const agentId = String(req.params.id);
  const agent = agents.get(agentId);
  if (!agent) return res.status(404).json({ error: 'Agent not found' });
  agent.state = 'terminated';
  broadcast(agent.id, 'terminated', agent);
  agents.delete(agentId);
  res.json({ ok: true });
});

// GET /api/agents/:id/stream — SSE
router.get('/:id/stream', (req: Request, res: Response) => {
  const agentId = String(req.params.id);
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
  });
  res.write(`event: connected\ndata: {"agentId":"${agentId}"}\n\n`);

  const send = (event: string, data: unknown) => {
    res.write(`event: ${event}\ndata: ${JSON.stringify(data)}\n\n`);
  };

  if (!agentStreams.has(agentId)) agentStreams.set(agentId, new Set());
  agentStreams.get(agentId)!.add(send);

  const keepalive = setInterval(() => res.write(':keepalive\n\n'), 15000);
  req.on('close', () => {
    clearInterval(keepalive);
    agentStreams.get(agentId)?.delete(send);
  });
});

export default router;
