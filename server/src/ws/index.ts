import { WebSocketServer, WebSocket } from 'ws';
import type { Server as HttpServer } from 'http';
import { subscribeToChannel, unsubscribeFromChannel, unsubscribeAll } from './hub.js';

interface WsClient {
  ws: WebSocket;
  lastPong: number;
}

const clients = new Map<WebSocket, WsClient>();
const HEARTBEAT_INTERVAL = 15_000;
const HEARTBEAT_TIMEOUT = 60_000;

export function setupWebSocket(server: HttpServer): WebSocketServer {
  const wss = new WebSocketServer({ server, path: '/ws' });

  const heartbeat = setInterval(() => {
    const now = Date.now();
    for (const [ws, client] of clients) {
      if (now - client.lastPong > HEARTBEAT_TIMEOUT) {
        ws.terminate();
        clients.delete(ws);
        unsubscribeAll(ws);
      }
    }
  }, HEARTBEAT_INTERVAL);

  wss.on('connection', (ws) => {
    clients.set(ws, { ws, lastPong: Date.now() });

    ws.on('message', (raw) => {
      try {
        const msg = JSON.parse(raw.toString());
        if (msg.type === 'pong' || msg.type === 'ping') {
          const client = clients.get(ws);
          if (client) client.lastPong = Date.now();
          if (msg.type === 'ping') ws.send(JSON.stringify({ type: 'pong' }));
          return;
        }

        if (msg.type === 'subscribe' && msg.channel) {
          subscribeToChannel(ws, msg.channel);
          ws.send(JSON.stringify({ type: 'subscribed', channel: msg.channel }));
          return;
        }

        if (msg.type === 'unsubscribe' && msg.channel) {
          unsubscribeFromChannel(ws, msg.channel);
          return;
        }
      } catch {
        // ignore malformed messages
      }
    });

    ws.on('close', () => {
      clients.delete(ws);
      unsubscribeAll(ws);
    });

    ws.on('error', () => {
      clients.delete(ws);
      unsubscribeAll(ws);
    });

    ws.send(JSON.stringify({ type: 'connected', heartbeatMs: HEARTBEAT_INTERVAL }));
  });

  wss.on('error', () => {});

  return wss;
}
