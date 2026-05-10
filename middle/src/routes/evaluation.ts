import { Router, type Request, type Response } from 'express';
import { getEngine } from '../engine.js';

const router = Router();

interface EvalResult {
  id: string;
  benchmark: string;
  model: string;
  score: number;
  metric: string;
  timestamp: number;
  metadata?: Record<string, unknown>;
}

const results: EvalResult[] = [];

router.get('/results', (_req: Request, res: Response) => {
  res.json({ results, total: results.length });
});

router.get('/results/:benchmark', (req: Request, res: Response) => {
  const filtered = results.filter(r => r.benchmark === req.params.benchmark);
  res.json({ benchmark: req.params.benchmark, results: filtered, total: filtered.length });
});

router.post('/results', (req: Request, res: Response) => {
  const { benchmark, model, score, metric, metadata } = req.body;
  if (!benchmark || !model || score === undefined) {
    return res.status(400).json({ error: 'benchmark, model, and score required' });
  }
  const entry: EvalResult = {
    id: `eval_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`,
    benchmark,
    model,
    score: Number(score),
    metric: metric || 'accuracy',
    timestamp: Date.now(),
    metadata,
  };
  results.push(entry);
  const engine = getEngine();
  engine.appendActivity('eval.result', true, `Eval ${benchmark} for ${model}: ${score}`);

  if (results.length > 10000) results.shift();
  res.status(201).json({ result: entry });
});

router.get('/summary', (_req: Request, res: Response) => {
  const byBenchmark: Record<string, { best: number; latest: number; count: number }> = {};
  for (const r of results) {
    if (!byBenchmark[r.benchmark]) byBenchmark[r.benchmark] = { best: -Infinity, latest: 0, count: 0 };
    byBenchmark[r.benchmark].count++;
    byBenchmark[r.benchmark].latest = r.score;
    if (r.score > byBenchmark[r.benchmark].best) byBenchmark[r.benchmark].best = r.score;
  }
  res.json({ summary: byBenchmark });
});

export default router;
