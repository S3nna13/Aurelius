import { Router } from 'express'
import { getEngine } from '../engine.js'
import { requireScope } from '../middleware/auth.js'

const router = Router()

router.get('/', (req, res) => {
  const engine = getEngine()
  const limit = req.query.limit ? parseInt(req.query.limit as string, 10) : undefined
  const entries = engine.getActivity(limit)
  res.json({ entries })
})

router.post('/', requireScope('activity:write'), (req, res) => {
  const engine = getEngine()
  const { command, success, output } = req.body || {}
  if (!command) {
    res.status(400).json({ error: 'Command required' })
    return
  }
  const entry = engine.appendActivity(command, success !== false, output || '')
  res.json({ success: true, entry })
})

export default router
