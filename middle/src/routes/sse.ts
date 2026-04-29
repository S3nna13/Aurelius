import { Router } from 'express'
import { getEngine } from '../engine.js'
import { requireScope, requireAdmin } from '../middleware/auth.js'

const router = Router()
const MAX_SSE_CLIENTS = 50

interface SSEClient {
  id: string
  res: import('express').Response
  userId?: string
}

const clients = new Map<string, SSEClient>()
let clientIdCounter = 0

function broadcast(event: string, data: unknown): void {
  const msg = `event: ${event}\ndata: ${JSON.stringify(data)}\n\n`
  for (const [, client] of clients) {
    try {
      client.res.write(msg)
    } catch {
      clients.delete(client.id)
    }
  }
}

router.get('/events', requireScope('sse:read'), (req, res) => {
  if (clients.size >= MAX_SSE_CLIENTS) {
    res.status(429).json({ error: 'Too many SSE connections' })
    return
  }
  const clientId = `sse-${++clientIdCounter}`

  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
    'X-Accel-Buffering': 'no',
  })

  const engine = getEngine()
  res.write(`event: connected\ndata: ${JSON.stringify({ clientId, timestamp: Date.now() })}\n\n`)

  const heartbeat = setInterval(() => {
    try {
      const stats = engine.getNotificationStats()
      const activity = engine.getActivity(5)
      res.write(`event: heartbeat\ndata: ${JSON.stringify({ stats, recentActivity: activity, timestamp: Date.now() })}\n\n`)
    } catch {
      clearInterval(heartbeat)
    }
  }, 15000)

  clients.set(clientId, { id: clientId, res, userId: req.user?.id })

  req.on('close', () => {
    clearInterval(heartbeat)
    clients.delete(clientId)
  })
})

router.post('/events/broadcast', requireScope('sse:broadcast'), requireAdmin, (req, res) => {
  const { event, data } = req.body || {}
  if (!event || !data) {
    res.status(400).json({ error: 'Event and data required' })
    return
  }
  if (!/^[a-zA-Z0-9_-]+$/.test(String(event))) {
    res.status(400).json({ error: 'Invalid event name' })
    return
  }
  broadcast(event, data)
  res.json({ success: true, clients: clients.size })
})

export { broadcast }
export default router
