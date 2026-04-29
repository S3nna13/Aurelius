import { Router, type Request, type Response } from 'express';
import { loadRegistrySnapshot } from '../registry_bridge.js';

const router = Router();

function categoryFilter<T extends { category: string }>(items: T[], category?: string): T[] {
  if (!category) {
    return items;
  }
  return items.filter((item) => item.category === category);
}

// GET /api/registry/agents — live agent registry from the Python layer
router.get('/agents', async (req: Request, res: Response) => {
  try {
    const snapshot = await loadRegistrySnapshot();
    const category = typeof req.query.category === 'string' ? req.query.category : undefined;
    const agents = categoryFilter(snapshot.agents, category);
    res.json({
      agents,
      total: agents.length,
      categories: snapshot.agent_categories,
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Failed to load live agent registry'
    res.status(502).json({
      error: 'Registry unavailable',
      message,
    });
  }
});

// GET /api/registry/agents/:id
router.get('/agents/:id', async (req: Request, res: Response) => {
  try {
    const snapshot = await loadRegistrySnapshot();
    const agent = snapshot.agents.find((item) => item.id === req.params.id);
    if (!agent) {
      res.status(404).json({ error: 'Agent type not found' });
      return;
    }
    res.json({ agent });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Failed to load live agent registry'
    res.status(502).json({
      error: 'Registry unavailable',
      message,
    });
  }
});

// GET /api/registry/skills — live skills registry from the Python layer
router.get('/skills', async (req: Request, res: Response) => {
  try {
    const snapshot = await loadRegistrySnapshot();
    const category = typeof req.query.category === 'string' ? req.query.category : undefined;
    const skills = categoryFilter(snapshot.skills, category);
    res.json({
      skills,
      total: skills.length,
      categories: snapshot.skill_categories,
    });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Failed to load live skills registry'
    res.status(502).json({
      error: 'Registry unavailable',
      message,
    });
  }
});

// GET /api/registry/skills/:id
router.get('/skills/:id', async (req: Request, res: Response) => {
  try {
    const snapshot = await loadRegistrySnapshot();
    const skill = snapshot.skills.find((item) => item.id === req.params.id);
    if (!skill) {
      res.status(404).json({ error: 'Skill not found' });
      return;
    }
    res.json({ skill });
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Failed to load live skills registry'
    res.status(502).json({
      error: 'Registry unavailable',
      message,
    });
  }
});

export default router;
