import { Router } from 'express'
import { config } from '../config.js'

const router = Router()

router.get('/', async (_req, res) => {
  try {
    const upstreamRes = await fetch(`${config.upstreamUrl}/v1/models`)
    if (upstreamRes.ok) {
      const data = await upstreamRes.json()
      res.json(data)
      return
    }
  } catch {
    // fallback to default
  }
  res.json({
    object: 'list',
    data: [{ id: 'aurelius', object: 'model', created: Math.floor(Date.now() / 1000), owned_by: 'aurelius' }],
  })
})

export default router
