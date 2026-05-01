import { WebSocketServer, WebSocket } from 'ws';
import type { Server as HttpServer } from 'http';
import { subscribeToChannel, unsubscribeFromChannel, unsubscribeAll, broadcastToChannel } from './hub.js';
import { getEngine } from '../config.js';

interface WsClient {
  ws: WebSocket;
  lastPong: number;
}

const clients = new Map<WebSocket, WsClient>();
const HEARTBEAT_INTERVAL = 15_000;
const HEARTBEAT_TIMEOUT = 60_000;

function handleChatMessage(ws: WebSocket, text: string): void {
  const engine = getEngine();
  const responseText = `Echo: ${text}`;
  const words = responseText.split(' ');

  // Stream tokens with delays to simulate real inference
  let i = 0;
  const stream = () => {
    if (i >= words.length) {
      ws.send(JSON.stringify({ type: 'chat_done', payload: { text: responseText } }));
      engine.appendActivity(text, true, responseText);
      broadcastToChannel('system', { type: 'activity', command: text, success: true });
      return;
    }
    const token = (i > 0 ? ' ' : '') + words[i];
    ws.send(JSON.stringify({ type: 'chat_token', payload: { token } }));
    i++;
    const delay = 30 + Math.random() * 20; // 30-50ms per token
    setTimeout(stream, delay);
  };
  stream();
}

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

        // Heartbeat
        if (msg.type === 'pong' || msg.type === 'ping') {
          const client = clients.get(ws);
          if (client) client.lastPong = Date.now();
          if (msg.type === 'ping') ws.send(JSON.stringify({ type: 'pong' }));
          return;
        }

        // Channel subscribe
        if (msg.type === 'subscribe' && msg.payload?.channel) {
          subscribeToChannel(ws, msg.payload.channel);
          ws.send(JSON.stringify({ type: 'subscribed', payload: { channel: msg.payload.channel } }));
          return;
        }

        // Channel unsubscribe
        if (msg.type === 'unsubscribe' && msg.payload?.channel) {
          unsubscribeFromChannel(ws, msg.payload.channel);
          return;
        }

        // Chat message — stream response token by token
        if (msg.type === 'chat' && msg.payload?.text) {
          handleChatMessage(ws, msg.payload.text);
          return;
        }

        // Legacy subscribe (for backward compat with old clients)
        if (msg.type === 'subscribe' && msg.channel) {
          subscribeToChannel(ws, msg.channel);
          ws.send(JSON.stringify({ type: 'subscribed', channel: msg.channel }));
          return;
        }

        // Legacy unsubscribe
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

    ws.send(JSON.stringify({ type: 'connected', payload: { heartbeatMs: HEARTBEAT_INTERVAL } }));
  });

  wss.on('error', () => {});

  return wss;
}
