import type { Request, Response } from 'express';
import { getEngine } from '../config.js';

export function listWorkflows(_req: Request, res: Response): void {
  const engine = getEngine();
  const workflows = engine.listWorkflows();
  const running = workflows.filter((w: any) => w.status === 'running').length;
  const completed = workflows.filter((w: any) => w.status === 'completed').length;
  const failed = workflows.filter((w: any) => w.status === 'failed').length;
  res.json({ workflows, summary: { total: workflows.length, running, completed, failed } });
}

export function workflowDetail(req: Request, res: Response): void {
  const wf = getEngine().getWorkflow(req.params.id as string);
  if (!wf) {
    res.status(404).json({ error: `Workflow ${req.params.id as string} not found` });
    return;
  }
  res.json(wf);
}

export function triggerWorkflow(req: Request, res: Response): void {
  const { trigger } = req.body;
  const id = req.params.id as string;
  if (!trigger || typeof trigger !== 'string') {
    res.status(400).json({ error: 'trigger is required' });
    return;
  }
  const engine = getEngine();
  const status = trigger === 'start' ? 'running' : trigger === 'complete' ? 'completed' : trigger === 'fail' ? 'failed' : 'pending';
  engine.updateWorkflowStatus(id, status);
  engine.appendLog('INFO', 'workflows', `Workflow ${id} -> ${status}`);
  res.json({ success: true, workflow_id: id, trigger, state: status });
}
