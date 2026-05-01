import { Router } from 'express'
import { getEngine } from '../engine.js'

const router = Router()

router.get('/health', (_req, res) => {
  res.json({
    status: 'ok',
    version: getEngine().getConfig('app.version') || '0.1.0',
    uptime: process.uptime(),
  })
})

router.get('/healthz', (_req, res) => {
  res.json({ alive: true })
})

router.get('/readyz', (_req, res) => {
  try {
    getEngine().listAgents()
    res.json({ ready: true })
  } catch {
    res.status(503).json({ ready: false })
  }
})

export default router
