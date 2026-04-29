import { WebSocketServer, type WebSocket } from 'ws'
import type { Server } from 'http'
import { getEngine } from '../engine.js'
import { joinRoom, leaveRoom, broadcastToRoom, broadcastToAll, getRoomList } from './rooms.js'
import { validateApiKey, type AuthUser } from '../middleware/auth.js'

interface WsMessage {
  type: string
  payload?: Record<string, unknown>
}

interface AuthenticatedSocket extends WebSocket {
  authUser?: AuthUser
}

const clients = new Set<AuthenticatedSocket>()
const MAX_WS_MESSAGE_SIZE = 65536
const WS_COMMANDS_REQUIRING_ADMIN = new Set(['command', 'agent:terminate', 'config:update'])

export function setupWebSocket(server: Server): WebSocketServer {
  const wss = new WebSocketServer({ server, path: '/ws', maxPayload: MAX_WS_MESSAGE_SIZE })

  wss.on('connection', (ws: AuthenticatedSocket, req) => {
    const apiKey = req.headers['x-api-key'] || req.headers['authorization']
    const key = typeof apiKey === 'string'
      ? apiKey.replace(/^Bearer\s+/i, '').trim()
      : null
    if (!key || !validateApiKey(key)) {
      ws.close(1008, 'Unauthorized')
      return
    }
    ws.authUser = validateApiKey(key)

    clients.add(ws)

    const engine = getEngine()
    ws.send(JSON.stringify({
      type: 'connected',
      payload: {
        agents: engine.listAgents(),
        stats: engine.getNotificationStats(),
        rooms: getRoomList(),
        timestamp: Date.now(),
      },
    }))

    ws.on('message', (raw) => {
      const rawStr = raw.toString()
      if (rawStr.length > MAX_WS_MESSAGE_SIZE) {
        ws.send(JSON.stringify({ type: 'error', payload: { message: 'Message too large' } }))
        return
      }
      try {
        const msg = JSON.parse(rawStr) as WsMessage
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

function handleMessage(ws: AuthenticatedSocket, msg: WsMessage): void {
  const engine = getEngine()
  const user = ws.authUser

  if (!user) {
    ws.send(JSON.stringify({ type: 'error', payload: { message: 'Not authenticated' } }))
    return
  }

  if (WS_COMMANDS_REQUIRING_ADMIN.has(msg.type) && user.role !== 'admin' && !user.scopes.includes('*')) {
    ws.send(JSON.stringify({ type: 'error', payload: { message: 'Admin access required' } }))
    return
  }

  switch (msg.type) {
    case 'ping':
      ws.send(JSON.stringify({ type: 'pong', payload: { timestamp: Date.now() } }))
      break

    case 'subscribe': {
      const { room } = msg.payload || {}
      if (room && typeof room === 'string') {
        joinRoom(room, ws)
        ws.send(JSON.stringify({ type: 'subscribed', payload: { room } }))
      }
      break
    }

    case 'unsubscribe': {
      const { room } = msg.payload || {}
      if (room && typeof room === 'string') {
        leaveRoom(room, ws)
        ws.send(JSON.stringify({ type: 'unsubscribed', payload: { room } }))
      }
      break
    }

    case 'room:message': {
      const { room, data } = msg.payload || {}
      if (room && typeof room === 'string') {
        const count = broadcastToRoom(room, { type: 'room:message', payload: { room, data, timestamp: Date.now() } })
        ws.send(JSON.stringify({ type: 'room:delivered', payload: { room, count } }))
      }
      break
    }

    case 'get:rooms':
      ws.send(JSON.stringify({ type: 'rooms', payload: { rooms: getRoomList() } }))
      break

    case 'get:agents':
      ws.send(JSON.stringify({ type: 'agents', payload: { agents: engine.listAgents() } }))
      break

    case 'get:activity':
      ws.send(JSON.stringify({ type: 'activity', payload: { entries: engine.getActivity(50) } }))
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
        ws.send(JSON.stringify({ type: 'command:ack', payload: { command, timestamp: Date.now() } }))
        broadcastToAll({ type: 'activity:new', payload: { command, timestamp: Date.now() } })
      }
      break
    }

    default:
      ws.send(JSON.stringify({ type: 'error', payload: { message: `Unknown type: ${msg.type}` } }))
  }
}

export function broadcastNotification(notification: unknown): void {
  const msg = JSON.stringify({ type: 'notification', payload: notification })
  for (const ws of clients) {
    if (ws.readyState === ws.OPEN) {
      try { ws.send(msg) } catch { /* best effort */ }
    }
  }
}

export function broadcastActivity(activity: unknown): void {
  const msg = JSON.stringify({ type: 'activity:new', payload: activity })
  for (const ws of clients) {
    if (ws.readyState === ws.OPEN) {
      try { ws.send(msg) } catch { /* best effort */ }
    }
  }
}
