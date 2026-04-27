import { WebSocket } from 'ws'

interface Room {
  name: string
  clients: Set<WebSocket>
  metadata: Record<string, unknown>
  createdAt: number
}

const rooms = new Map<string, Room>()

export function createRoom(name: string, metadata: Record<string, unknown> = {}): Room {
  if (rooms.has(name)) return rooms.get(name)!
  const room: Room = { name, clients: new Set(), metadata, createdAt: Date.now() }
  rooms.set(name, room)
  return room
}

export function deleteRoom(name: string): boolean {
  return rooms.delete(name)
}

export function joinRoom(roomName: string, ws: WebSocket): Room {
  const room = createRoom(roomName)
  room.clients.add(ws)
  ws.on('close', () => leaveRoom(roomName, ws))
  return room
}

export function leaveRoom(roomName: string, ws: WebSocket): void {
  const room = rooms.get(roomName)
  if (!room) return
  room.clients.delete(ws)
  if (room.clients.size === 0) {
    rooms.delete(roomName)
  }
}

export function broadcastToRoom(roomName: string, data: unknown): number {
  const room = rooms.get(roomName)
  if (!room) return 0
  const msg = JSON.stringify(data)
  let count = 0
  for (const ws of room.clients) {
    if (ws.readyState === ws.OPEN) {
      ws.send(msg)
      count++
    }
  }
  return count
}

export function broadcastToAll(data: unknown): number {
  const msg = JSON.stringify(data)
  let count = 0
  for (const [, room] of rooms) {
    for (const ws of room.clients) {
      if (ws.readyState === ws.OPEN) {
        ws.send(msg)
        count++
      }
    }
  }
  return count
}

export function getRoomList(): Array<{ name: string; clients: number; metadata: Record<string, unknown>; createdAt: number }> {
  const list: Array<{ name: string; clients: number; metadata: Record<string, unknown>; createdAt: number }> = []
  for (const [, room] of rooms) {
    list.push({ name: room.name, clients: room.clients.size, metadata: room.metadata, createdAt: room.createdAt })
  }
  return list
}

export function getRoom(name: string): Room | undefined {
  return rooms.get(name)
}

export function getRoomCount(): number {
  return rooms.size
}

export function getClientCount(): number {
  let total = 0
  for (const [, room] of rooms) total += room.clients.size
  return total
}
