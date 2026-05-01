import { useEffect, useRef, useCallback, useState } from 'react'

type MessageHandler = (data: unknown) => void

interface UseWebSocketOptions {
  url?: string
  autoReconnect?: boolean
  reconnectInterval?: number
  maxReconnects?: number
  onStatusChange?: (connected: boolean) => void
}

interface UseWebSocketReturn {
  connected: boolean
  send: (type: string, payload?: Record<string, unknown>) => void
  on: (type: string, handler: MessageHandler) => void
  off: (type: string, handler: MessageHandler) => void
  lastMessage: unknown
  subscribe: (channel: string) => void
  unsubscribe: (channel: string) => void
  onChannel: (channel: string, handler: MessageHandler) => () => void
}

export function useWebSocket(opts: UseWebSocketOptions = {}): UseWebSocketReturn {
  const {
    url = `ws://${window.location.host}/ws`,
    autoReconnect = true,
    reconnectInterval = 3000,
    maxReconnects = 10,
    onStatusChange,
  } = opts

  const wsRef = useRef<WebSocket | null>(null)
  const handlersRef = useRef<Map<string, Set<MessageHandler>>>(new Map())
  const reconnectCountRef = useRef(0)
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const [connected, setConnected] = useState(false)
  const [lastMessage, setLastMessage] = useState<unknown>(null)
  const onStatusChangeRef = useRef(onStatusChange)
  const connectRef = useRef<() => void>(() => {})
  useEffect(() => {
    onStatusChangeRef.current = onStatusChange
  }, [onStatusChange])

  const updateConnected = useCallback((v: boolean) => {
    setConnected(v)
    onStatusChangeRef.current?.(v)
  }, [])

  const scheduleReconnect = useCallback(() => {
    if (autoReconnect && reconnectCountRef.current < maxReconnects) {
      reconnectCountRef.current += 1
      reconnectTimerRef.current = setTimeout(() => {
        connectRef.current()
      }, reconnectInterval)
    }
  }, [autoReconnect, maxReconnects, reconnectInterval])

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return
    try {
      const ws = new WebSocket(url)
      wsRef.current = ws
      ws.onopen = () => { updateConnected(true); reconnectCountRef.current = 0 }
      ws.onmessage = (event) => {
        try {
          const data = JSON.parse(event.data) as { type: string; payload?: unknown }
          setLastMessage(data)
          if (data.type === 'connected' || data.type === 'pong') return
          const handlers = handlersRef.current.get(data.type)
          if (handlers) { for (const handler of handlers) handler(data.payload) }
        } catch { /* ignore */ }
      }
      ws.onclose = () => {
        updateConnected(false); wsRef.current = null
        scheduleReconnect()
      }
      ws.onerror = () => { ws.close() }
    } catch {
      scheduleReconnect()
    }
  }, [url, updateConnected, scheduleReconnect])

  useEffect(() => {
    connectRef.current = connect
  }, [connect])

  useEffect(() => {
    connect()
    return () => { clearTimeout(reconnectTimerRef.current); wsRef.current?.close() }
  }, [connect])

  const send = useCallback((type: string, payload?: Record<string, unknown>) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify({ type, payload }))
    }
  }, [])

  const on = useCallback((type: string, handler: MessageHandler) => {
    if (!handlersRef.current.has(type)) handlersRef.current.set(type, new Set())
    handlersRef.current.get(type)!.add(handler)
  }, [])

  const off = useCallback((type: string, handler: MessageHandler) => {
    handlersRef.current.get(type)?.delete(handler)
  }, [])

  const subscribe = useCallback((channel: string) => { send('subscribe', { channel }) }, [send])
  const unsubscribe = useCallback((channel: string) => { send('unsubscribe', { channel }) }, [send])
  const onChannel = useCallback((channel: string, handler: MessageHandler) => {
    on(channel, handler)
    return () => off(channel, handler)
  }, [on, off])

  return { connected, send, on, off, lastMessage, subscribe, unsubscribe, onChannel }
}
