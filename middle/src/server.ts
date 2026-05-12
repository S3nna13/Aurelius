import express from 'express'
import cors from 'cors'
import { createServer } from 'http'
import { config } from './config.js'
import { authMiddleware, requireAdmin } from './middleware/auth.js'
import { rateLimiter } from './middleware/rate-limiter.js'
import { requestLogger } from './middleware/logger.js'
import { errorHandler, notFoundHandler } from './middleware/error-handler.js'
import { setupWebSocket } from './ws/handler.js'
import { getEngine } from './engine.js'
import healthRoutes from './routes/health.js'
import agentsRoutes from './routes/agents.js'
import activityRoutes from './routes/activity.js'
import notificationsRoutes from './routes/notifications.js'
import configRoutes from './routes/config.js'
import memoryRoutes from './routes/memory.js'
import logsRoutes from './routes/logs.js'
import chatRoutes from './routes/chat.js'
import modelsRoutes from './routes/models.js'
import pluginsRoutes from './routes/plugins.js'
import authRoutes from './routes/auth.js'
import filesRoutes from './routes/files.js'
import sseRoutes from './routes/sse.js'
import statsRoutes from './routes/stats.js'
import systemRoutes from './routes/system.js'
import schedulerRoutes from './routes/scheduler.js'
import searchRoutes from './routes/search.js'
import tracesRoutes from './routes/traces.js'
import registryRoutes from './routes/registry.js'
import brainRoutes from './routes/brain.js'
import evaluationRoutes from './routes/evaluation.js'
import ragRoutes from './routes/rag.js'
import { ProviderRouter } from './provider_router.js'

const providerRouter = new ProviderRouter();

export function buildApp() {
  const app = express()

  app.use(cors({ origin: config.corsOrigin, credentials: true }))
  app.use(express.json({ limit: '50mb' }))
  app.use(requestLogger)
  app.use(authMiddleware)
  app.use(rateLimiter)

  // Public routes — exempted by auth middleware
  app.use('/api', healthRoutes)
  app.use(healthRoutes)
  app.use('/api/auth', authRoutes)

  // Protected routes — mounted after auth middleware
  app.use('/api/sse', sseRoutes)
  app.use('/api/agents', agentsRoutes)
  app.use('/api/activity', activityRoutes)
  app.use('/api/notifications', notificationsRoutes)
  app.use('/api/config', configRoutes)
  app.use('/api/memory', memoryRoutes)
  app.use('/api/logs', logsRoutes)
  app.use('/api/chat', chatRoutes)
  app.use('/api/models', modelsRoutes)
  app.use('/api/plugins', pluginsRoutes)
  app.use('/api/files', filesRoutes)
  app.use('/api/stats', statsRoutes)
  app.use('/api/system', systemRoutes)
  app.use('/api/scheduler', schedulerRoutes)
  app.use('/api/search', searchRoutes)
  app.use('/api/traces', tracesRoutes)
  app.use('/api/registry', registryRoutes)
  app.use('/api/brain', brainRoutes)
  app.use('/api/evaluation', evaluationRoutes)
  app.use('/api/rag', ragRoutes)

  // Skills routes — backed by data engine
  app.get('/api/skills', (_req, res) => {
    const engine = getEngine()
    const skills = engine.listSkills() as Array<{ id: string; name: string; active: boolean }>
    res.json({
      skills,
      total: skills.length,
      enabled: skills.filter((s: { active: boolean }) => s.active).length,
      totalTools: 0,
      totalSkills: skills.length,
    })
  })
  app.get('/api/skills/:id', (req, res) => {
    const engine = getEngine()
    const skill = engine.getSkill(String(req.params.id))
    if (!skill) return res.status(404).json({ error: 'Skill not found' })
    res.json({ skill })
  })
  app.post('/api/skills/execute', async (req, res) => {
    const { skillId, params } = req.body || {}
    if (!skillId) return res.status(400).json({ error: 'skillId required' })
    const engine = getEngine()
    const skill = engine.getSkill(String(skillId))
    if (!skill) return res.status(404).json({ error: 'Skill not found' })
    engine.appendActivity('skill.execute', true, `Skill ${skillId} triggered`)
    try {
      const result = await providerRouter.complete({
        model: 'aurelius-1.3b',
        messages: [
          { role: 'system', content: `Execute skill: ${JSON.stringify(skill)}` },
          { role: 'user', content: JSON.stringify(params || {}) },
        ],
        max_tokens: 1024,
      })
      res.json({ success: true, skillId, output: result?.choices?.[0]?.message?.content || '' })
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'execution failed'
      res.status(502).json({ error: msg })
    }
  })
  // Workflows routes — backed by data engine
  app.get('/api/workflows', (_req, res) => {
    const engine = getEngine()
    const workflows = engine.listWorkflows() as Array<{ id: string; name: string; status: string }>
    const summary = { total: 0, running: 0, completed: 0, failed: 0 }
    for (const w of workflows) {
      summary.total++
      if (w.status === 'running') summary.running++
      else if (w.status === 'completed') summary.completed++
      else if (w.status === 'failed') summary.failed++
    }
    res.json({ workflows, summary })
  })
  app.get('/api/workflows/:id', (req, res) => {
    const engine = getEngine()
    const workflow = engine.getWorkflow(String(req.params.id))
    if (!workflow) return res.status(404).json({ error: 'Workflow not found' })
    res.json({ workflow })
  })
  app.post('/api/workflows/:id/trigger', async (req, res) => {
    const engine = getEngine()
    const id = String(req.params.id)
    const workflow = engine.getWorkflow(id)
    if (!workflow) return res.status(404).json({ error: 'Workflow not found' })
    engine.updateWorkflowStatus(id, 'running')
    engine.appendActivity('workflow.trigger', true, `Workflow ${id} triggered`)
    res.json({ success: true, workflowId: id, status: 'running' })
  })
  app.get('/api/status', (_req, res) => res.json({ status: 'ok', uptime: process.uptime() }))

  // Command dispatch — route command to agent LLM and return result
  app.post('/api/command', requireAdmin, async (req, res) => {
    const { agentId, command } = req.body || {}
    if (!command) {
      res.status(400).json({ error: 'command required' })
      return
    }
    const engine = getEngine()
    const agentState = agentId ? (engine.listAgents() as Array<{ id: string; role: string; state: string }>).find(a => a.id === agentId) : null
    if (agentId && !agentState) {
      res.status(404).json({ error: 'Agent not found' })
      return
    }

    const systemPrompt = agentState
      ? `You are ${agentState.id}, an AI agent with role: ${agentState.role}. Respond to the following command concisely.`
      : 'You are an AI assistant. Respond to the following command concisely.'

    const messages = [
      { role: 'system', content: systemPrompt },
      { role: 'user', content: command },
    ]

    try {
      const result = await providerRouter.complete({ model: 'aurelius-1.3b', messages, max_tokens: 1024 })
      const responseText = result?.choices?.[0]?.message?.content || 'No response generated'

      engine.appendActivity('command.dispatch', true, `Command "${command.slice(0, 80)}" sent to ${agentId || 'system'}`)

      res.json({
        success: true,
        agentId: agentId || null,
        command,
        response: responseText,
        dispatchedAt: new Date().toISOString(),
      })
    } catch (err: unknown) {
      const msg = err instanceof Error ? err.message : 'dispatch failed'
      engine.appendActivity('command.error', false, `Command failed: ${msg}`)
      res.status(502).json({ success: false, error: msg })
    }
  })

  // OpenAI-compatible completions proxy for frontend
  app.post('/api/v1/completions', async (req, res) => {
    try {
      const headers: Record<string, string> = { 'Content-Type': 'application/json' }
      if (config.serviceApiKey) {
        headers['Authorization'] = `Bearer ${config.serviceApiKey}`
        headers['X-Api-Key'] = config.serviceApiKey
      }
      const upstreamRes = await fetch(`${config.upstreamUrl}/v1/chat/completions`, {
        method: 'POST',
        headers,
        body: JSON.stringify(req.body),
      })
      const data = await upstreamRes.json()
      res.status(upstreamRes.status).json(data)
    } catch {
      res.status(502).json({ error: 'Upstream unavailable' })
    }
  })

  // Provider router — unified Aurelius + OpenAI
  app.get('/api/provider/status', (_req, res) => {
    res.json({
      primary: 'aurelius',
      fallback: 'openai',
      openai_configured: !!providerRouter.openaiApiKey,
      stats: providerRouter.getStats(),
      models: providerRouter.getAvailableModels(),
    });
  });

  // Error handling
  app.use(errorHandler)
  app.use(notFoundHandler)

  return app
}

export function startServer() {
  const app = buildApp()
  const server = createServer(app)
  const wss = setupWebSocket(server)

  server.listen(config.port, config.host, () => {
    console.log(`[middle] BFF server listening on http://${config.host}:${config.port}`)
    console.log(`[middle] WebSocket available at ws://${config.host}:${config.port}/ws`)
    console.log(`[middle] Proxying upstream to ${config.upstreamUrl}`)
  })

  const shutdown = () => {
    console.log('[middle] Shutting down gracefully...')
    wss.close()
    server.close(() => process.exit(0))
  }

  process.on('SIGTERM', shutdown)
  process.on('SIGINT', shutdown)

  return { app, server, wss }
}
