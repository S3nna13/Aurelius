import { Router } from 'express'
import { config } from '../config.js'

const router = Router()

router.get('/', async (_req, res) => {
  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (config.serviceApiKey) {
    headers['Authorization'] = `Bearer ${config.serviceApiKey}`
    headers['X-Api-Key'] = config.serviceApiKey
  }
  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), 10000)
  try {
    const upstreamRes = await fetch(`${config.upstreamUrl}/v1/models`, { headers, signal: controller.signal })
    clearTimeout(timer)
    const data = await upstreamRes.json()
    res.status(upstreamRes.status).json(data)
  } catch (error: unknown) {
    clearTimeout(timer)
    const message = error instanceof Error ? error.message : 'Upstream unavailable'
    res.status(502).json({ error: 'Upstream unavailable', message })
  }
})

export default router
