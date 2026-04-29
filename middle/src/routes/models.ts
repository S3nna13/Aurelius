import { Router } from 'express'
import { config } from '../config.js'

const router = Router()

router.get('/', async (_req, res) => {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (config.serviceApiKey) {
    headers['Authorization'] = `Bearer ${config.serviceApiKey}`
    headers['X-Api-Key'] = config.serviceApiKey
  }
  try {
    const upstreamRes = await fetch(`${config.upstreamUrl}/v1/models`, { headers })
    const data = await upstreamRes.json()
    res.status(upstreamRes.status).json(data)
  } catch (error: unknown) {
    const message = error instanceof Error ? error.message : 'Upstream unavailable'
    res.status(502).json({ error: 'Upstream unavailable', message })
  }
})

export default router
