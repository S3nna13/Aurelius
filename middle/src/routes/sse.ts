import { Router } from 'express'
import { getEngine } from '../engine.js'

const router = Router()

interface SSEClient {
  id: string
  res: import('express').Response
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

router.get('/events', (req, res) => {
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

  clients.set(clientId, { id: clientId, res })

  req.on('close', () => {
    clearInterval(heartbeat)
    clients.delete(clientId)
  })
})

router.post('/events/broadcast', (req, res) => {
  const { event, data } = req.body || {}
  if (!event || !data) {
    res.status(400).json({ error: 'Event and data required' })
    return
  }
  broadcast(event, data)
  res.json({ success: true, clients: clients.size })
})

export { broadcast }
export default router
