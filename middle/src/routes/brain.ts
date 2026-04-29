import { Router, Request, Response } from 'express';
import { requireScope, requireAdmin } from '../middleware/auth.js';
import { spawn } from 'child_process';
import path from 'path';
import { fileURLToPath } from 'url';

const router = Router();
let brainConcurrency = 0;
const BRAIN_TIMEOUT_MS = 30000;
const MAX_BRAIN_CONCURRENCY = 1;
const MAX_INPUT_LENGTH = 4096;
const repoRoot = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '../../..');

const ALLOWED_CHARS = /^[a-zA-Z0-9_\-\.\/\s\{\}\[\]\":,\(\)]+$/;

function sanitizeInput(input: unknown): string {
  if (typeof input !== 'string') {
    throw new Error('Input must be a string');
  }
  if (input.length > MAX_INPUT_LENGTH) {
    throw new Error(`Input exceeds maximum length of ${MAX_INPUT_LENGTH}`);
  }
  if (!ALLOWED_CHARS.test(input)) {
    throw new Error('Input contains disallowed characters');
  }
  return input.replace(/[;|&$`\\]/g, '');
}

function runBrain(cmd: string, res: Response): void {
  if (brainConcurrency >= MAX_BRAIN_CONCURRENCY) {
    res.status(429).json({ error: 'Brain execution busy, try again later' })
    return
  }
  brainConcurrency++

  const python = spawn('python3', ['-c', cmd], { cwd: repoRoot, stdio: ['pipe', 'pipe', 'pipe'] })
  let result = ''
  let stderr = ''
  let killed = false

  const timer = setTimeout(() => {
    killed = true
    python.kill()
    brainConcurrency = Math.max(0, brainConcurrency - 1)
    res.status(504).json({ error: 'Brain execution timed out' })
  }, BRAIN_TIMEOUT_MS)

  python.stdout.on('data', (data: Buffer) => { result += data.toString() })
  python.stderr.on('data', (data: Buffer) => { stderr += data.toString() })
  python.on('close', (code: number) => {
    clearTimeout(timer)
    if (killed) return
    brainConcurrency = Math.max(0, brainConcurrency - 1)
    if (code === 0) {
      try { res.json(JSON.parse(result)) }
      catch { res.json({ error: 'Parse error', output: result.slice(0, 2000) }) }
    } else {
      res.status(500).json({ error: 'Brain process failed', code, stderr: stderr.slice(0, 1000) })
    }
  })
  python.on('error', (err) => {
    clearTimeout(timer)
    brainConcurrency = Math.max(0, brainConcurrency - 1)
    if (!res.headersSent) {
      res.status(500).json({ error: 'Failed to spawn brain process', details: err.message })
    }
  })
}

// POST /api/brain/think — run the neural brain on input
router.post('/think', requireScope('brain:execute'), requireAdmin, (req: Request, res: Response) => {
  const { input } = req.body;
  if (!input) { res.status(400).json({ error: 'input required' }); return }

  let safeInput: string;
  try {
    safeInput = sanitizeInput(input);
  } catch (err: unknown) {
    res.status(400).json({ error: err instanceof Error ? err.message : 'Invalid input' });
    return;
  }

  const cmd = `
import json, sys
sys.path.insert(0, '.')
from aurelius.neural_brain import NeuralBrain
brain = NeuralBrain()
ctx = brain.run(${JSON.stringify(safeInput)})
print(json.dumps({
    "state": ctx.state,
    "plan": ctx.plan,
    "reasoning_steps": len(ctx.reasoning),
    "actions": len(ctx.actions),
    "verifications": len(ctx.verifications),
    "reflections": len(ctx.reflections),
    "output": ctx.output[:500],
    "stats": brain.get_stats(),
}))`;
  runBrain(cmd, res);
});

// GET /api/brain/stats — brain usage statistics
router.get('/stats', requireScope('brain:execute'), (_req: Request, res: Response) => {
  const cmd = `
import json, sys
sys.path.insert(0, '.')
from aurelius.neural_brain import NeuralBrain
brain = NeuralBrain()
print(json.dumps({"stats": brain.get_stats(), "message": "Brain initialized. Run POST /api/brain/think with input."}))`;
  runBrain(cmd, res);
});

// POST /api/upgrade/run — run a self-upgrade cycle
router.post('/upgrade/run', requireScope('brain:execute'), requireAdmin, (_req: Request, res: Response) => {
  const cmd = `
import json, sys
sys.path.insert(0, '.')
from aurelius.self_upgrade import SelfUpgradeSystem
upgrader = SelfUpgradeSystem()
upgrader.record_metric("eval_accuracy", 0.72, target=0.90)
upgrader.record_metric("training_loss", 2.1, target=1.5)
upgrader.record_metric("inference_latency", 350, target=100, unit="ms")
proposal = upgrader.run_upgrade_cycle()
print(json.dumps(upgrader.get_summary()))`;
  runBrain(cmd, res);
});

// GET /api/upgrade/status — upgrade system status
router.get('/upgrade/status', (_req: Request, res: Response) => {
  res.json({
    system: 'Self-Upgrade Layer',
    status: 'active',
    cycle_count: 0,
    last_improvement: null,
    metrics_tracked: ['eval_accuracy', 'training_loss', 'inference_latency', 'safety_score'],
  });
});

export default router;
