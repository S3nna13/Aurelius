import { Router } from 'express'
import { getEngine } from '../engine.js'

const router = Router()

router.get('/', (_req, res) => {
  const engine = getEngine()
  const config = engine.getAllConfig()
  res.json({ config })
})

router.post('/', (req, res) => {
  const engine = getEngine()
  const { config } = req.body || {}
  if (!config || typeof config !== 'object') {
    res.status(400).json({ error: 'Config object required' })
    return
  }
  for (const [key, value] of Object.entries(config)) {
    engine.setConfig(key, String(value))
  }
  engine.appendActivity('config.update', true, 'Configuration updated')
  res.json({ success: true, config: engine.getAllConfig() })
})

router.get('/:key', (req, res) => {
  const engine = getEngine()
  const value = engine.getConfig(req.params.key)
  if (value === null) {
    res.status(404).json({ error: 'Config key not found' })
    return
  }
  res.json({ key: req.params.key, value })
})

router.put('/:key', (req, res) => {
  const engine = getEngine()
  const { value } = req.body || {}
  if (value === undefined) {
    res.status(400).json({ error: 'Value required' })
    return
  }
  engine.setConfig(req.params.key, String(value))
  engine.appendActivity('config.update', true, `${req.params.key} updated`)
  res.json({ success: true })
})

export default router
