import { Router } from 'express'
import { getEngine } from '../engine.js'

const router = Router()

router.get('/', (_req, res) => {
  const engine = getEngine()
  const agents = engine.listAgents()
  res.json({
    agents,
    counts: {
      agents_online: agents.filter((a) => a.state === 'active' || a.state === 'running').length,
      agents_total: agents.length,
    },
  })
})

router.get('/:id', (req, res) => {
  const engine = getEngine()
  const agent = engine.getAgent(req.params.id)
  if (!agent) {
    res.status(404).json({ error: 'Agent not found' })
    return
  }
  res.json(agent)
})

router.post('/:id/command', (req, res) => {
  const engine = getEngine()
  const { command } = req.body || {}
  if (!command) {
    res.status(400).json({ error: 'Command required' })
    return
  }
  const agent = engine.getAgent(req.params.id)
  if (!agent) {
    res.status(404).json({ error: 'Agent not found' })
    return
  }
  const output = `[${agent.id}] executed: ${command}`
  const entry = engine.appendActivity(`agent.${agent.id}`, true, output)
  res.json({ success: true, output, activity: entry })
})

router.post('/:id/state', (req, res) => {
  const engine = getEngine()
  const { state } = req.body || {}
  if (!state) {
    res.status(400).json({ error: 'State required' })
    return
  }
  const agent = engine.getAgent(req.params.id)
  if (!agent) {
    res.status(404).json({ error: 'Agent not found' })
    return
  }
  engine.upsertAgent(agent.id, state, agent.role, agent.metricsJson)
  engine.appendActivity(`agent.${agent.id}.state`, true, `State changed to ${state}`)
  res.json({ success: true, state })
})

export default router
