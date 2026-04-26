import type { Request, Response } from 'express';

const MODES = [
  { id: 'default', name: 'Default', description: 'Balanced mode for general use.', allowed_tools: ['read', 'search', 'list'], response_style: 'balanced' },
  { id: 'supervised', name: 'Supervised', description: 'Actions require human approval.', allowed_tools: ['read', 'search'], response_style: 'cautious' },
  { id: 'autonomous', name: 'Autonomous', description: 'Full autonomy for trusted tasks.', allowed_tools: ['read', 'search', 'write', 'execute'], response_style: 'direct' },
];

export function listModes(_req: Request, res: Response): void {
  res.json({ modes: MODES });
}
