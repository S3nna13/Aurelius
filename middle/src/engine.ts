import { DataEngine } from 'aurelius-data-engine'
import type {
  AgentState, ActivityEntry, Notification, NotificationStats,
  MemoryLayer, MemoryEntry, LogRecord,
} from 'aurelius-data-engine'

let _engine: DataEngine | null = null

export function getEngine(): DataEngine {
  if (!_engine) {
    _engine = new DataEngine()
  }
  return _engine
}

export function seedInitialData(): void {
  const engine = getEngine()

  const agents = [
    { id: 'hermes', state: 'active', role: 'notification-router' },
    { id: 'openclaw', state: 'active', role: 'task-orchestrator' },
    { id: 'cerebrum', state: 'idle', role: 'memory-manager' },
    { id: 'vigil', state: 'active', role: 'security-warden' },
    { id: 'thoth', state: 'idle', role: 'code-analyst' },
  ]

  for (const a of agents) {
    if (!engine.getAgent(a.id)) {
      engine.upsertAgent(a.id, a.state, a.role, JSON.stringify({ uptime: 0, tasks: 0, memory: '128MB' }))
    }
  }

  engine.appendActivity('system.boot', true, 'Aurelius Mission Control initialized')
  engine.appendActivity('system.auth', true, 'Admin session authenticated')
  engine.appendActivity('agent.hermes.start', true, 'Hermes notification router started')

  engine.addNotification('system', 'high', 'startup', 'System Ready', 'All agents initialized successfully')
  engine.addNotification('system', 'medium', 'info', 'Data Engine Active', 'Rust data engine is managing state')

  engine.setConfig('app.version', '0.1.0')
  engine.setConfig('app.name', 'Aurelius')

  engine.appendLog('info', 'system', 'System boot sequence complete')
  engine.appendLog('info', 'engine', 'Data engine initialized with seed data')
}

export {
  DataEngine,
  AgentState, ActivityEntry, Notification,
  NotificationStats, MemoryLayer, MemoryEntry, LogRecord,
}
