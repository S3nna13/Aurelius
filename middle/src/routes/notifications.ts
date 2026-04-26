import { Router } from 'express'
import { getEngine } from '../engine.js'

const router = Router()

router.get('/', (req, res) => {
  const engine = getEngine()
  const { category, priority, read, limit } = req.query
  const notifications = engine.getNotifications(
    category as string | undefined,
    priority as string | undefined,
    read !== undefined ? read === 'true' : undefined,
    limit ? parseInt(limit as string, 10) : undefined,
  )
  res.json({ notifications })
})

router.get('/stats', (_req, res) => {
  const engine = getEngine()
  res.json(engine.getNotificationStats())
})

router.post('/', (req, res) => {
  const engine = getEngine()
  const { channel, priority, category, title, body } = req.body || {}
  if (!title) {
    res.status(400).json({ error: 'Title required' })
    return
  }
  const notification = engine.addNotification(
    channel || 'system',
    priority || 'medium',
    category || 'info',
    title,
    body || '',
  )
  res.json({ success: true, notification })
})

router.post('/:id/read', (req, res) => {
  const engine = getEngine()
  const ok = engine.markNotificationRead(req.params.id)
  res.json({ success: ok })
})

router.post('/read-all', (req, res) => {
  const engine = getEngine()
  const { category } = req.body || {}
  const count = engine.markAllNotificationsRead(category || undefined)
  res.json({ success: true, markedRead: count })
})

export default router
