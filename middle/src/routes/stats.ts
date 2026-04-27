import { Router } from 'express'
import { getEngine } from '../engine.js'

const router = Router()

router.get('/', (_req, res) => {
  const engine = getEngine()
  const stats = engine.getStats()
  const activity = engine.getActivity(1000)

  const successRate = activity.length > 0
    ? Math.round((activity.filter((a) => a.success).length / activity.length) * 100)
    : 100

  const activityByHour = new Array(24).fill(0)
  for (const a of activity) {
    const hour = new Date(a.timestamp * 1000).getHours()
    activityByHour[hour]++
  }

  const topCommands = new Map<string, number>()
  for (const a of activity) {
    topCommands.set(a.command, (topCommands.get(a.command) || 0) + 1)
  }
  const topCommandsSorted = [...topCommands.entries()]
    .sort((a, b) => b[1] - a[1])
    .slice(0, 10)
    .map(([command, count]) => ({ command, count }))

  res.json({
    ...stats,
    successRate,
    totalRequests: stats.activityCount,
    activityByHour,
    topCommands: topCommandsSorted,
    uptimeSeconds: process.uptime(),
  })
})

router.get('/activity-hourly', (_req, res) => {
  const engine = getEngine()
  const activity = engine.getActivity(10000)
  const byHour = new Array(24).fill(0)
  for (const a of activity) {
    const hour = new Date(a.timestamp * 1000).getHours()
    byHour[hour]++
  }
  res.json({ hourly: byHour.map((count, hour) => ({ hour, count })) })
})

router.get('/commands', (_req, res) => {
  const engine = getEngine()
  const activity = engine.getActivity(10000)
  const cmdMap = new Map<string, { count: number; success: number; fail: number }>()
  for (const a of activity) {
    const entry = cmdMap.get(a.command) || { count: 0, success: 0, fail: 0 }
    entry.count++
    if (a.success) entry.success++
    else entry.fail++
    cmdMap.set(a.command, entry)
  }
  const commands = [...cmdMap.entries()]
    .sort((a, b) => b[1].count - a[1].count)
    .map(([command, stats]) => ({ command, ...stats, successRate: Math.round((stats.success / stats.count) * 100) }))
  res.json({ commands })
})

export default router
