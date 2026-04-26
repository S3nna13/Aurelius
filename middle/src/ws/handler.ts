import { WebSocketServer, type WebSocket } from 'ws'
import type { Server } from 'http'
import { getEngine } from '../engine.js'
import { registerApiKey, type AuthUser } from '../middleware/auth.js'

interface WsMessage {
  type: string
  payload?: Record<string, unknown>
}

const clients = new Set<WebSocket>()

function broadcast(data: unknown): void {
  const msg = JSON.stringify(data)
  for (const ws of clients) {
    if (ws.readyState === ws.OPEN) {
      ws.send(msg)
    }
  }
}

export function setupWebSocket(server: Server): WebSocketServer {
  const wss = new WebSocketServer({ server, path: '/ws' })

  wss.on('connection', (ws, req) => {
    clients.add(ws)

    const engine = getEngine()
    ws.send(JSON.stringify({
      type: 'connected',
      payload: {
        agents: engine.listAgents(),
        stats: engine.getNotificationStats(),
        timestamp: Date.now(),
      },
    }))

    ws.on('message', (raw) => {
      try {
        const msg = JSON.parse(raw.toString()) as WsMessage
        handleMessage(ws, msg)
      } catch {
        ws.send(JSON.stringify({ type: 'error', payload: { message: 'Invalid JSON' } }))
      }
    })

    ws.on('close', () => {
      clients.delete(ws)
    })

    ws.on('error', () => {
      clients.delete(ws)
    })
  })

  return wss
}

function handleMessage(ws: WebSocket, msg: WsMessage): void {
  const engine = getEngine()

  switch (msg.type) {
    case 'ping':
      ws.send(JSON.stringify({ type: 'pong', payload: { timestamp: Date.now() } }))
      break

    case 'get:agents':
      ws.send(JSON.stringify({ type: 'agents', payload: { agents: engine.listAgents() } }))
      break

    case 'get:activity':
      ws.send(JSON.stringify({
        type: 'activity',
        payload: { entries: engine.getActivity(50) },
      }))
      break

    case 'get:notifications':
      ws.send(JSON.stringify({
        type: 'notifications',
        payload: {
          notifications: engine.getNotifications(undefined, undefined, undefined, 50),
          stats: engine.getNotificationStats(),
        },
      }))
      break

    case 'get:status':
      ws.send(JSON.stringify({
        type: 'status',
        payload: {
          agents: engine.listAgents(),
          activity: engine.getActivity(10),
          notifications: engine.getNotificationStats(),
          memory: engine.getMemoryLayers(),
        },
      }))
      break

    case 'command': {
      const { command } = msg.payload || {}
      if (command) {
        engine.appendActivity('ws.command', true, String(command))
        ws.send(JSON.stringify({
          type: 'command:ack',
          payload: { command, timestamp: Date.now() },
        }))
        broadcast({
          type: 'activity:new',
          payload: { command, timestamp: Date.now() },
        })
      }
      break
    }

    default:
      ws.send(JSON.stringify({ type: 'error', payload: { message: `Unknown type: ${msg.type}` } }))
  }
}

export function broadcastNotification(notification: unknown): void {
  broadcast({ type: 'notification', payload: notification })
}

export function broadcastActivity(activity: unknown): void {
  broadcast({ type: 'activity:new', payload: activity })
}
