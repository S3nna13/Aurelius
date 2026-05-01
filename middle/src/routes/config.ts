import { Router } from 'express'
import { getEngine } from '../engine.js'

const PROTECTED_CONFIG_KEYS = new Set(['api_key', 'license_key', 'license_activated', 'require_auth'])

const router = Router()

router.get('/', (_req, res) => {
  const engine = getEngine()
  const raw = engine.getAllConfig()
  const config = Object.fromEntries(
    Object.entries(raw).filter(([k]) => !PROTECTED_CONFIG_KEYS.has(k))
  )
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
    if (PROTECTED_CONFIG_KEYS.has(key)) {
      res.status(403).json({ error: `Config key '${key}' is protected and cannot be set via API` })
      return
    }
    engine.setConfig(key, String(value))
  }
  engine.appendActivity('config.update', true, 'Configuration updated')
  const raw = engine.getAllConfig()
  const safeConfig = Object.fromEntries(
    Object.entries(raw).filter(([k]) => !PROTECTED_CONFIG_KEYS.has(k))
  )
  res.json({ success: true, config: safeConfig })
})

router.get('/:key', (req, res) => {
  const engine = getEngine()
  if (PROTECTED_CONFIG_KEYS.has(req.params.key)) {
    res.status(403).json({ error: `Config key '${req.params.key}' is protected` })
    return
  }
  const value = engine.getConfig(req.params.key)
  if (value === null) {
    res.status(404).json({ error: 'Config key not found' })
    return
  }
  res.json({ key: req.params.key, value })
})

router.put('/:key', (req, res) => {
  const engine = getEngine()
  if (PROTECTED_CONFIG_KEYS.has(req.params.key)) {
    res.status(403).json({ error: `Config key '${req.params.key}' is protected` })
    return
  }
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
