import { Router } from 'express'
import { getEngine } from '../engine.js'
import { config } from '../config.js'

const router = Router()

router.post('/command', async (req, res) => {
  const engine = getEngine()
  const { command } = req.body || {}
  if (!command) {
    res.status(400).json({ error: 'Command required' })
    return
  }

  engine.appendActivity('chat.command', true, command)

  try {
    const controller = new AbortController()
    const timeoutId = setTimeout(() => controller.abort(), 30_000)
    const upstreamRes = await fetch(`${config.upstreamUrl}/v1/chat/completions`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model: 'aurelius',
        messages: [{ role: 'user', content: command }],
        max_tokens: 1024,
      }),
      signal: controller.signal,
    })
    clearTimeout(timeoutId)

    if (!upstreamRes.ok) {
      const text = await upstreamRes.text()
      res.json({
        success: false,
        error: `Upstream error: ${upstreamRes.status} - ${text}`,
      })
      return
    }

    const data = await upstreamRes.json() as { choices?: Array<{ message?: { content?: string } }> }
    const reply = data.choices?.[0]?.message?.content || 'No response from model'

    engine.appendActivity('chat.response', true, `Response: ${reply.slice(0, 100)}...`)

    res.json({ success: true, output: reply })
  } catch (err) {
    const msg = err instanceof Error ? err.message : 'Connection failed'
    res.json({ success: false, error: msg })
  }
})

router.get('/status', (_req, res) => {
  const engine = getEngine()
  res.json({
    agents: engine.listAgents(),
    memory: engine.getMemoryLayers(),
  })
})

router.get('/events', (req, res) => {
  res.writeHead(200, {
    'Content-Type': 'text/event-stream',
    'Cache-Control': 'no-cache',
    Connection: 'keep-alive',
  })

  const timer = setInterval(() => {
    const engine = getEngine()
    const stats = engine.getNotificationStats()
    const recentActivity = engine.getActivity(5)
    res.write(`data: ${JSON.stringify({ type: 'heartbeat', stats, recentActivity })}\n\n`)
  }, 15000)

  req.on('close', () => {
    clearInterval(timer)
  })
})

router.get('/suggestions', (_req, res) => {
  res.json({
    suggestions: [
      'Show system status',
      'List all active agents',
      'Run diagnostic check',
      'Show recent activity',
      'What skills are available?',
      'Analyze memory usage',
      'Check notification history',
      'Display workflow status',
    ],
  })
})

export default router
