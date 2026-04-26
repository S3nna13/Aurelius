import type { Request, Response } from 'express';
import { getEngine } from '../config.js';

export function listSkills(_req: Request, res: Response): void {
  res.json({ skills: getEngine().listSkills() });
}

export function skillDetail(req: Request, res: Response): void {
  const skill = getEngine().getSkill(req.params.id as string);
  if (!skill) {
    res.status(404).json({ error: `Skill ${req.params.id as string} not found` });
    return;
  }
  res.json(skill);
}

export function executeSkill(req: Request, res: Response): void {
  const { skill_id } = req.body;
  if (!skill_id || typeof skill_id !== 'string') {
    res.status(400).json({ error: 'skill_id is required' });
    return;
  }
  const engine = getEngine();
  engine.appendLog('INFO', 'skills', `Executed skill: ${skill_id}`);
  res.json({ success: true, output: `Executed ${skill_id}`, duration_ms: 0 });
}
