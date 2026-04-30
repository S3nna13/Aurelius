import type { Request, Response } from 'express';
import { getEngine } from '../config.js';
import { broadcastToChannel } from '../ws/hub.js';

export function listNotifications(req: Request, res: Response): void {
  const { category, priority, read, limit } = req.query;
  const notifications = getEngine().getNotifications(
    category as string | undefined,
    priority as string | undefined,
    read !== undefined ? read === 'true' : undefined,
    limit ? parseInt(limit as string, 10) : undefined,
  );
  res.json({ notifications });
}

export function notificationStats(_req: Request, res: Response): void {
  res.json(getEngine().getNotificationStats());
}

export function createNotification(req: Request, res: Response): void {
  const { channel, priority, category, title, body } = req.body;
  if (!title || !body) {
    res.status(400).json({ error: 'title and body are required' });
    return;
  }
  const engine = getEngine();
  const notif = engine.addNotification(
    channel || 'system',
    priority || 'normal',
    category || 'info',
    title,
    body,
  );
  engine.appendLog('INFO', 'notifications', `Notification: ${title}`);
  broadcastToChannel('notifications', { type: 'new', notification: notif });
  res.json({ success: true, notification: notif });
}

export function markNotificationRead(req: Request, res: Response): void {
  const { id } = req.body;
  const engine = getEngine();
  const success = engine.markNotificationRead(id);
  if (success) {
    broadcastToChannel('notifications', { type: 'read', id });
  }
  res.json({ success });
}

export function markAllNotificationsRead(req: Request, res: Response): void {
  const category = req.body?.category || undefined;
  const engine = getEngine();
  const count = engine.markAllNotificationsRead(category);
  broadcastToChannel('notifications', { type: 'read-all', category: category || null, count });
  res.json({ success: true, count });
}

export function getNotifPrefs(_req: Request, res: Response): void {
  const raw = getEngine().getConfig('notification_preferences');
  const preferences = raw ? JSON.parse(raw) : {};
  res.json({ preferences });
}

export function setNotifPrefs(req: Request, res: Response): void {
  const { preferences } = req.body;
  if (!preferences || typeof preferences !== 'object') {
    res.status(400).json({ error: 'preferences must be an object' });
    return;
  }
  getEngine().setConfig('notification_preferences', JSON.stringify(preferences || {}));
  res.json({ success: true });
}
