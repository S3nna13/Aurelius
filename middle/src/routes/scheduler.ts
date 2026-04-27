import { Router } from 'express'

interface CronTask {
  id: string
  name: string
  cron: string
  command: string
  enabled: boolean
  lastRun: string | null
  lastSuccess: boolean | null
  nextRun: string | null
  createdAt: string
}

const router = Router()
const tasks = new Map<string, CronTask>()
const intervals = new Map<string, ReturnType<typeof setInterval>>()

function parseCron(expr: string): number | null {
  const parts = expr.trim().split(/\s+/)
  if (parts.length !== 5) return null
  const every = parts[0] === '*' && parts[1] === '*' && parts[2] === '*' && parts[3] === '*' && parts[4] === '*'
  if (every) return 60000
  const minutes = parseInt(parts[0], 10)
  if (!isNaN(minutes) && parts[1] === '*' && parts[2] === '*' && parts[3] === '*' && parts[4] === '*') {
    return minutes * 60000
  }
  const hours = parseInt(parts[1], 10)
  if (parts[0] === '*' && !isNaN(hours) && parts[2] === '*' && parts[3] === '*' && parts[4] === '*') {
    return hours * 3600000
  }
  return null
}

router.get('/', (_req, res) => {
  res.json({ tasks: Array.from(tasks.values()) })
})

router.post('/', (req, res) => {
  const { name, cron, command } = req.body || {}
  if (!name || !cron || !command) {
    res.status(400).json({ error: 'Name, cron, and command required' })
    return
  }

  const id = `task-${Date.now()}-${Math.random().toString(36).slice(2, 6)}`
  const task: CronTask = {
    id, name, cron, command, enabled: true,
    lastRun: null, lastSuccess: null, nextRun: null,
    createdAt: new Date().toISOString(),
  }

  tasks.set(id, task)
  scheduleTask(id, task)
  res.json({ success: true, task })
})

router.delete('/:id', (req, res) => {
  const id = req.params.id
  clearInterval(intervals.get(id))
  intervals.delete(id)
  tasks.delete(id)
  res.json({ success: true })
})

router.post('/:id/toggle', (req, res) => {
  const task = tasks.get(req.params.id)
  if (!task) {
    res.status(404).json({ error: 'Task not found' })
    return
  }
  task.enabled = !task.enabled
  if (task.enabled) scheduleTask(task.id, task)
  else clearInterval(intervals.get(task.id))
  res.json({ success: true, enabled: task.enabled })
})

function scheduleTask(id: string, task: CronTask) {
  clearInterval(intervals.get(id))
  const ms = parseCron(task.cron)
  if (!ms) return
  const interval = setInterval(async () => {
    if (!task.enabled) return
    task.lastRun = new Date().toISOString()
    try {
      const res = await fetch(`http://localhost:${process.env.MIDDLE_PORT || 3001}/api/command`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ command: task.command }),
      })
      const data = await res.json()
      task.lastSuccess = data.success
    } catch {
      task.lastSuccess = false
    }
  }, ms)
  intervals.set(id, interval)
}

export default router
