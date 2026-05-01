import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import { loadConfig, getEngine } from './config.js';
import { authMiddleware, validateAuthConfig, rateLimitMiddleware, requestLoggerMiddleware, metricsMiddleware, metricsHandler, errorHandler } from './middleware/index.js';
import { registerRoutes } from './routes/index.js';
import { spaFallback } from './spa.js';
import { setupWebSocket } from './ws/index.js';
import { broadcastToChannel } from './ws/hub.js';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export function createApp() {
  const config = loadConfig();
  const engine = getEngine();

  validateAuthConfig(config);

  const app = express();

  const corsOrigins = (process.env.CORS_ORIGINS || (process.env.NODE_ENV === 'production' ? '' : 'http://localhost:5173,http://localhost:3000'))
    .split(',')
    .map(s => s.trim())
    .filter(Boolean);
  app.use(cors({
    origin: corsOrigins.length > 0 ? corsOrigins : false,
    ...(corsOrigins.length > 0 && { credentials: true }),
  }));
  app.use(requestLoggerMiddleware());
  app.use(express.json({ limit: '1mb' }));
  app.use(metricsMiddleware);
  app.use(authMiddleware(config));
  app.use(rateLimitMiddleware(config));

  app.get('/metrics', metricsHandler);

  const frontendDist = path.resolve(__dirname, '../../frontend/dist');
  app.use('/assets', express.static(path.join(frontendDist, 'assets'), { maxAge: '1y', immutable: true }));
  app.use(express.static(frontendDist));

  const apiRouter = express.Router();
  registerRoutes(apiRouter);
  app.use('/api', apiRouter);

  app.use(spaFallback);

  app.use(errorHandler);

  const seed = () => {
    if (engine.listAgents().length === 0) {
      engine.upsertAgent('hermes', 'ACTIVE', 'Notification Router', '{"messages_routed":1240,"uptime_pct":99.8,"latency_ms":12}');
      engine.upsertAgent('openclaw', 'ACTIVE', 'Task Orchestrator', '{"tasks_completed":342,"uptime_pct":99.5,"latency_ms":45}');
      engine.upsertAgent('cerebrum', 'ACTIVE', 'Memory Manager', '{"queries_served":8901,"uptime_pct":99.9,"latency_ms":28}');
      engine.upsertAgent('vigil', 'IDLE', 'Security Warden', '{"alerts_processed":567,"uptime_pct":99.7,"latency_ms":8}');

      engine.appendActivity('System initialized', true, 'All agents registered');
      engine.appendLog('INFO', 'gateway', 'Aurelius gateway started');

      broadcastToChannel('agents', { type: 'full', agents: engine.listAgents() });
      broadcastToChannel('system', { type: 'info', message: 'System initialized' });
    }
  };
  seed();

  return { app, config, engine };
}

export function startServer() {
  const { app, config } = createApp();
  const server = app.listen(config.port, config.host, () => {
    console.log(`Aurelius Gateway — http://localhost:${config.port}`);
  });
  setupWebSocket(server);
  return server;
}
