import { Router } from 'express'
import { getEngine } from '../engine.js'

const router = Router()

router.get('/', (req, res) => {
  const engine = getEngine()
  const { q, type, limit: limitStr } = req.query
  const query = (q as string || '').trim()
  const limit = limitStr ? parseInt(limitStr as string, 10) : 20

  if (!query) {
    res.status(400).json({ error: 'Query parameter q is required' })
    return
  }

  const typeFilter = (type as string) || 'all'
  const results: Record<string, unknown> = {}

  if (typeFilter === 'all' || typeFilter === 'activity') {
    results.activity = engine.searchActivity(query, limit)
  }

  if (typeFilter === 'all' || typeFilter === 'logs') {
    results.logs = engine.searchLogs(query, undefined, limit)
  }

  if (typeFilter === 'all' || typeFilter === 'memory') {
    results.memory = engine.getMemoryEntries(undefined, query, limit)
  }

  if (typeFilter === 'all' || typeFilter === 'agents') {
    const allAgents = engine.listAgents()
    results.agents = allAgents.filter((a) =>
      a.id.toLowerCase().includes(query.toLowerCase()) ||
      a.role.toLowerCase().includes(query.toLowerCase())
    )
  }

  if (typeFilter === 'all' || typeFilter === 'notifications') {
    const allNotifs = engine.getNotifications(undefined, undefined, undefined, limit * 2)
    results.notifications = allNotifs.filter((n) =>
      n.title.toLowerCase().includes(query.toLowerCase()) ||
      n.body.toLowerCase().includes(query.toLowerCase())
    ).slice(0, limit)
  }

  if (typeFilter === 'all' || typeFilter === 'config') {
    const allConfig = engine.getAllConfig()
    const matched: Record<string, string> = {}
    for (const [key, value] of Object.entries(allConfig)) {
      if (key.toLowerCase().includes(query.toLowerCase()) || value.toLowerCase().includes(query.toLowerCase())) {
        matched[key] = value
      }
    }
    results.config = matched
  }

  res.json({
    query,
    type: typeFilter,
    results,
    total: Object.values(results).reduce((sum: number, arr: unknown) => sum + (Array.isArray(arr) ? arr.length : Object.keys(arr as Record<string, unknown>).length), 0),
  })
})

router.get('/suggestions', (_req, res) => {
  const engine = getEngine()
  const activity = engine.getActivity(100)
  const commands = new Set<string>()
  for (const a of activity) {
    commands.add(a.command)
  }
  res.json({
    suggestions: [
      ...Array.from(commands).slice(0, 10),
      'system', 'agent', 'config', 'error', 'memory', 'notification',
    ],
  })
})

export default router
