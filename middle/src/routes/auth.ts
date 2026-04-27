import { Router } from 'express'
import { registerApiKey, type AuthUser } from '../middleware/auth.js'
import { validateBody } from '../middleware/validation.js'
import { v4 as uuidv4 } from 'uuid'

const router = Router()

interface User {
  id: string
  username: string
  role: 'admin' | 'user'
  apiKeys: string[]
  createdAt: string
}

const users = new Map<string, User>()
const userSessions = new Map<string, { token: string; expiresAt: number }>()

users.set('admin', {
  id: 'user-admin',
  username: 'admin',
  role: 'admin',
  apiKeys: ['dev-key'],
  createdAt: new Date().toISOString(),
})

registerApiKey('dev-key', { id: 'user-admin', role: 'admin', scopes: ['*'] })

router.post('/login', validateBody([
  { field: 'apiKey', type: 'string', required: true, min: 1 },
]), (req, res) => {
  const { apiKey } = req.body

  for (const [, user] of users) {
    if (user.apiKeys.includes(apiKey)) {
      const token = uuidv4()
      const expiresAt = Date.now() + 86400000
      userSessions.set(token, { token, expiresAt })

      registerApiKey(apiKey, { id: user.id, role: user.role, scopes: ['*'] })

      res.json({
        success: true,
        token,
        tokenType: 'bearer',
        expiresIn: 86400,
        user: { id: user.id, username: user.username, role: user.role },
      })
      return
    }
  }

  res.status(401).json({ error: 'Invalid API key' })
})

router.post('/register', validateBody([
  { field: 'username', type: 'string', required: true, min: 3, max: 32 },
]), (req, res) => {
  const { username } = req.body

  for (const [, user] of users) {
    if (user.username === username) {
      res.status(409).json({ error: 'Username already taken' })
      return
    }
  }

  const id = `user-${uuidv4().slice(0, 8)}`
  const apiKey = `ak-${uuidv4().replace(/-/g, '')}`

  const user: User = {
    id,
    username,
    role: 'user',
    apiKeys: [apiKey],
    createdAt: new Date().toISOString(),
  }

  users.set(id, user)
  registerApiKey(apiKey, { id, role: 'user', scopes: ['read', 'write'] })

  res.json({
    success: true,
    user: { id, username, role: 'user' },
    apiKey,
  })
})

router.get('/users', (_req, res) => {
  const userList = Array.from(users.values()).map((u) => ({
    id: u.id,
    username: u.username,
    role: u.role,
    apiKeys: u.apiKeys.length,
    createdAt: u.createdAt,
  }))
  res.json({ users: userList })
})

router.post('/keys/generate', (req, res) => {
  const userId = req.user?.id
  if (!userId) {
    res.status(401).json({ error: 'Unauthorized' })
    return
  }

  const user = users.get(userId)
  if (!user) {
    res.status(404).json({ error: 'User not found' })
    return
  }

  const apiKey = `ak-${uuidv4().replace(/-/g, '')}`
  user.apiKeys.push(apiKey)
  registerApiKey(apiKey, { id: userId, role: user.role, scopes: ['*'] })

  res.json({ success: true, apiKey })
})

router.get('/keys', (req, res) => {
  const userId = req.user?.id
  if (!userId) {
    res.status(401).json({ error: 'Unauthorized' })
    return
  }

  const user = users.get(userId)
  if (!user) {
    res.status(404).json({ error: 'User not found' })
    return
  }

  res.json({ keys: user.apiKeys.map((k) => ({
    prefix: k.slice(0, 8) + '...',
    full: k,
  })) })
})

router.delete('/keys/:prefix', (req, res) => {
  const userId = req.user?.id
  if (!userId) {
    res.status(401).json({ error: 'Unauthorized' })
    return
  }

  const user = users.get(userId)
  if (!user) {
    res.status(404).json({ error: 'User not found' })
    return
  }

  const prefix = req.params.prefix
  const idx = user.apiKeys.findIndex((k) => k.startsWith(prefix))
  if (idx === -1) {
    res.status(404).json({ error: 'API key not found' })
    return
  }

  user.apiKeys.splice(idx, 1)
  res.json({ success: true })
})

export default router
