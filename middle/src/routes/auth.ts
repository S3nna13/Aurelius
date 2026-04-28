import { Router } from 'express'
import { registerApiKey, unregisterApiKey, type AuthUser } from '../middleware/auth.js'
import { validateBody } from '../middleware/validation.js'
import { v4 as uuidv4 } from 'uuid'
import { config } from '../config.js'

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
const inviteTokens = new Map<string, { role: 'admin' | 'user'; used: boolean }>()

const adminKey = process.env.AURELIUS_API_KEY || 'dev-key'
if (adminKey) {
  users.set('admin', {
    id: 'admin',
    username: 'admin',
    role: 'admin',
    apiKeys: [adminKey],
    createdAt: new Date().toISOString(),
  })
  registerApiKey(adminKey, { id: 'admin', role: 'admin', scopes: ['*'] })
}

router.post('/login', validateBody([
  { field: 'apiKey', type: 'string', required: true, min: 1 },
]), (req, res) => {
  const { apiKey } = req.body

  for (const [, user] of users) {
    if (user.apiKeys.includes(apiKey)) {
      const token = uuidv4()
      const expiresAt = Date.now() + 86400000
      userSessions.set(token, { token, expiresAt })

      registerApiKey(apiKey, { id: user.id, role: user.role, scopes: ['read', 'write'] })

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
  if (!config.allowPublicRegistration) {
    if (!req.user || req.user.role !== 'admin') {
      res.status(403).json({ error: 'Forbidden', message: 'Public registration is disabled' })
      return
    }
  }

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
  registerApiKey(apiKey, { id, role: 'user', scopes: ['read'] })

  // Mark invite token as used if provided
  const inviteToken = req.body.inviteToken as string | undefined
  if (inviteToken && inviteTokens.has(inviteToken)) {
    inviteTokens.get(inviteToken)!.used = true
  }

  res.json({
    success: true,
    user: { id, username, role: 'user' },
    apiKey,
  })
})

function cleanupExpiredSessions() {
  const now = Date.now()
  for (const [token, session] of userSessions) {
    if (session.expiresAt < now) {
      userSessions.delete(token)
    }
  }
}

// Run cleanup every 5 minutes
setInterval(cleanupExpiredSessions, 300_000)

router.get('/users', (_req, res) => {
  cleanupExpiredSessions()
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
  registerApiKey(apiKey, { id: userId, role: user.role, scopes: req.user?.scopes || ['read'] })

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

  const removedKey = user.apiKeys[idx]
  user.apiKeys.splice(idx, 1)
  unregisterApiKey(removedKey)
  res.json({ success: true })
})

export default router
