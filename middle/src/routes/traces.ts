import { Router, Request, Response } from 'express';
import { v4 as uuid } from 'uuid';
import { requireScope } from '../middleware/auth.js';

const router = Router();

interface TraceStep {
  id: string;
  type: 'thought' | 'tool_call' | 'tool_result' | 'action' | 'error' | 'observation';
  timestamp: number;
  content: string;
  metadata?: Record<string, unknown>;
  duration?: number;
}

interface Trace {
  id: string;
  agentId: string;
  agentName: string;
  task: string;
  status: 'running' | 'completed' | 'failed' | 'truncated';
  steps: TraceStep[];
  startedAt: number;
  completedAt?: number;
  totalDuration?: number;
  stepCount: number;
  tokenCount?: number;
}

const traces = new Map<string, Trace>();

// GET /api/traces
router.get('/', (req: Request, res: Response) => {
  const { agentId, status, limit } = req.query;
  let result = Array.from(traces.values());

  if (agentId) result = result.filter(t => t.agentId === agentId);
  if (status) result = result.filter(t => t.status === status);

  result.sort((a, b) => b.startedAt - a.startedAt);
  const capped = limit ? result.slice(0, Number(limit)) : result;

  res.json({
    traces: capped.map(({ steps, ...rest }) => ({
      ...rest,
      stepCount: steps.length,
      _steps: `${steps.length} steps`,
    })),
    total: result.length,
  });
});

// GET /api/traces/:id
router.get('/:id', (req: Request, res: Response) => {
  const traceId = String(req.params.id);
  const trace = traces.get(traceId);
  if (!trace) return res.status(404).json({ error: 'Trace not found' });
  res.json({ trace });
});

// POST /api/traces
router.post('/', requireScope('traces:write'), (req: Request, res: Response) => {
  const { agentId, agentName, task } = req.body;
  if (!agentId || !task) return res.status(400).json({ error: 'agentId and task required' });

  const trace: Trace = {
    id: uuid().slice(0, 12),
    agentId,
    agentName: agentName || 'unknown',
    task,
    status: 'running',
    steps: [],
    startedAt: Date.now(),
    stepCount: 0,
  };
  traces.set(trace.id, trace);
  if (traces.size > 10000) {
    const firstKey = traces.keys().next().value
    if (firstKey) traces.delete(firstKey)
  }
  res.status(201).json({ trace });
});

// POST /api/traces/:id/step
router.post('/:id/step', requireScope('traces:write'), (req: Request, res: Response) => {
  const traceId = String(req.params.id);
  const trace = traces.get(traceId);
  if (!trace) return res.status(404).json({ error: 'Trace not found' });

  const { type, content, metadata, duration } = req.body;
  if (!type || !content) return res.status(400).json({ error: 'type and content required' });

  const step: TraceStep = {
    id: uuid().slice(0, 8),
    type,
    timestamp: Date.now(),
    content,
    metadata,
    duration,
  };
  trace.steps.push(step);
  trace.stepCount = trace.steps.length;
  res.status(201).json({ step });
});

// PATCH /api/traces/:id
router.patch('/:id', requireScope('traces:write'), (req: Request, res: Response) => {
  const traceId = String(req.params.id);
  const trace = traces.get(traceId);
  if (!trace) return res.status(404).json({ error: 'Trace not found' });

  if (req.body.status) {
    trace.status = req.body.status;
    if (req.body.status === 'running') {
      delete trace.completedAt;
      delete trace.totalDuration;
    } else {
      trace.completedAt = Date.now();
      trace.totalDuration = trace.completedAt - trace.startedAt;
    }
  }
  if (req.body.tokenCount) trace.tokenCount = req.body.tokenCount;
  res.json({ trace });
});

export default router;
