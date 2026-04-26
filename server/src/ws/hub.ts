import type { WebSocket } from 'ws';

const channelSubs = new Map<string, Set<WebSocket>>();

export function subscribeToChannel(ws: WebSocket, channel: string): void {
  if (!channelSubs.has(channel)) channelSubs.set(channel, new Set());
  channelSubs.get(channel)!.add(ws);
}

export function unsubscribeFromChannel(ws: WebSocket, channel: string): void {
  channelSubs.get(channel)?.delete(ws);
}

export function unsubscribeAll(ws: WebSocket): void {
  for (const subs of channelSubs.values()) subs.delete(ws);
}

export function broadcastToChannel(channel: string, data: unknown): void {
  const msg = JSON.stringify({ channel, data });
  const subs = channelSubs.get(channel);
  if (!subs) return;
  for (const ws of subs) {
    if (ws.readyState === ws.OPEN) {
      ws.send(msg);
    }
  }
}

export function broadcastToChannels(channels: string[], data: unknown): void {
  for (const ch of channels) broadcastToChannel(ch, data);
}

export function getChannelSizes(): Record<string, number> {
  const sizes: Record<string, number> = {};
  for (const [ch, subs] of channelSubs) {
    sizes[ch] = subs.size;
  }
  return sizes;
}
