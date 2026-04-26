import { Router } from 'express'
import { getEngine } from '../engine.js'

const router = Router()

router.get('/', (req, res) => {
  const engine = getEngine()
  const { level, query, limit } = req.query
  const logs = engine.getLogs(
    level as string | undefined,
    query as string | undefined,
    limit ? parseInt(limit as string, 10) : undefined,
  )
  res.json({ logs })
})

router.post('/', (req, res) => {
  const engine = getEngine()
  const { level, logger, message } = req.body || {}
  if (!level || !message) {
    res.status(400).json({ error: 'Level and message required' })
    return
  }
  engine.appendLog(level, logger || 'api', message)
  res.json({ success: true })
})

export default router
