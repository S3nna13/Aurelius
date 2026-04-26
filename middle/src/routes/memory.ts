import { Router } from 'express'
import { getEngine } from '../engine.js'

const router = Router()

router.get('/layers', (_req, res) => {
  const engine = getEngine()
  const layers = engine.getMemoryLayers()
  res.json({ layers })
})

router.get('/entries', (req, res) => {
  const engine = getEngine()
  const { layer, query, limit } = req.query
  const entries = engine.getMemoryEntries(
    layer as string | undefined,
    query as string | undefined,
    limit ? parseInt(limit as string, 10) : undefined,
  )
  res.json({ entries })
})

router.post('/entries', (req, res) => {
  const engine = getEngine()
  const { layer, content } = req.body || {}
  if (!layer || !content) {
    res.status(400).json({ error: 'Layer and content required' })
    return
  }
  engine.addMemoryEntry(layer, content)
  engine.appendActivity('memory.add', true, `Entry added to ${layer}`)
  res.json({ success: true })
})

export default router
