import { Router, type Request, type Response } from 'express';
import { getEngine } from '../engine.js';

const router = Router();

interface Document {
  id: string;
  filename: string;
  content: string;
  chunkCount: number;
  uploadedAt: number;
  source: string;
}

const documents: Document[] = [];

router.get('/documents', (_req: Request, res: Response) => {
  res.json({
    documents: documents.map(d => ({ id: d.id, filename: d.filename, chunkCount: d.chunkCount, uploadedAt: d.uploadedAt, source: d.source })),
    total: documents.length,
  });
});

router.get('/documents/:id', (req: Request, res: Response) => {
  const doc = documents.find(d => d.id === req.params.id);
  if (!doc) return res.status(404).json({ error: 'Document not found' });
  res.json({ document: doc });
});

router.post('/documents', (req: Request, res: Response) => {
  const { filename, content, source } = req.body;
  if (!filename || !content) return res.status(400).json({ error: 'filename and content required' });

  const id = `doc_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
  const chunks = content.match(/[\s\S]{1,512}/g) || [];
  const doc: Document = { id, filename, content, chunkCount: chunks.length, uploadedAt: Date.now(), source: source || 'upload' };

  documents.push(doc);
  const engine = getEngine();
  engine.appendActivity('rag.upload', true, `Document ${filename} ingested (${chunks.length} chunks)`);

  for (let i = 0; i < chunks.length; i++) {
    engine.addMemoryEntry('rag', `[${id}:${i}] ${chunks[i]}`);
  }

  if (documents.length > 1000) documents.shift();
  res.status(201).json({ document: { id, filename, chunkCount: doc.chunkCount, uploadedAt: doc.uploadedAt } });
});

router.delete('/documents/:id', (req: Request, res: Response) => {
  const idx = documents.findIndex(d => d.id === req.params.id);
  if (idx === -1) return res.status(404).json({ error: 'Document not found' });
  documents.splice(idx, 1);
  res.json({ ok: true });
});

router.get('/search', (req: Request, res: Response) => {
  const query = String(req.query.q || '');
  if (!query) return res.status(400).json({ error: 'query parameter q required' });

  const results = documents
    .map(d => {
      const idx = d.content.toLowerCase().indexOf(query.toLowerCase());
      if (idx === -1) return null;
      const start = Math.max(0, idx - 100);
      const end = Math.min(d.content.length, idx + query.length + 100);
      return { documentId: d.id, filename: d.filename, snippet: d.content.slice(start, end), relevance: 1.0 };
    })
    .filter(Boolean);

  res.json({ query, results, total: results.length });
});

export default router;
