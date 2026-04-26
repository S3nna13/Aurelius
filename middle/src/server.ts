import express from 'express'
import cors from 'cors'
import { createServer } from 'http'
import { config } from './config.js'
import { authMiddleware } from './middleware/auth.js'
import { rateLimiter } from './middleware/rate-limiter.js'
import { requestLogger } from './middleware/logger.js'
import { setupWebSocket } from './ws/handler.js'
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

export function buildApp() {
  const app = express()

  app.use(cors({ origin: config.corsOrigin, credentials: true }))
  app.use(express.json({ limit: '1mb' }))
  app.use(requestLogger)
  app.use(authMiddleware)
  app.use(rateLimiter)

  app.use('/api', healthRoutes)
  app.use('/api/agents', agentsRoutes)
  app.use('/api/activity', activityRoutes)
  app.use('/api/notifications', notificationsRoutes)
  app.use('/api/config', configRoutes)
  app.use('/api/memory', memoryRoutes)
  app.use('/api/logs', logsRoutes)
  app.use('/api', chatRoutes)
  app.use('/api/v1/models', modelsRoutes)
  app.use('/api/plugins', pluginsRoutes)

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
