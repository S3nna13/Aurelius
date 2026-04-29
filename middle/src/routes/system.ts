import { Router } from 'express'
import { execSync } from 'child_process'
import { cpus, totalmem, freemem, networkInterfaces, uptime, hostname } from 'os'
import { existsSync, readFileSync } from 'fs'

const router = Router()

router.use((req, res, next) => {
  if (req.user?.role !== 'admin') {
    res.status(403).json({ error: 'Forbidden', message: 'Admin access required' })
    return
  }
  next()
})

function getCPUInfo() {
  const c = cpus()
  const model = c.length > 0 ? c[0].model : 'unknown'
  return { model, cores: c.length, speed: c[0]?.speed || 0 }
}

function getNodeInfo() {
  return {
    version: process.version,
    platform: process.platform,
    arch: process.arch,
    pid: process.pid,
    uptime: process.uptime(),
    memoryUsage: process.memoryUsage(),
    cwd: process.cwd(),
  }
}

function getGitInfo(): Record<string, string> {
  try {
    const hash = execSync('git rev-parse HEAD', { encoding: 'utf-8' }).trim()
    const branch = execSync('git rev-parse --abbrev-ref HEAD', { encoding: 'utf-8' }).trim()
    const message = execSync('git log -1 --format=%s', { encoding: 'utf-8' }).trim()
    return { hash, branch, message }
  } catch {
    return { hash: 'unknown', branch: 'unknown', message: 'unknown' }
  }
}

router.get('/', (_req, res) => {
  const node = getNodeInfo()
  const cpu = getCPUInfo()

  const networks = networkInterfaces()
  const ips: string[] = []
  for (const [, ifaces] of Object.entries(networks)) {
    if (ifaces) {
      for (const iface of ifaces) {
        if (iface.family === 'IPv4' && !iface.internal) ips.push(iface.address)
      }
    }
  }

  res.json({
    hostname: hostname(),
    ips,
    platform: node.platform,
    arch: node.arch,
    cpu,
    memory: {
      total: totalmem(),
      free: freemem(),
      used: totalmem() - freemem(),
      usagePercent: Math.round(((totalmem() - freemem()) / totalmem()) * 100),
    },
    node: node.version,
    pid: node.pid,
    processUptime: node.uptime,
    osUptime: uptime(),
    memoryUsage: node.memoryUsage,
    git: getGitInfo(),
  })
})

router.get('/env', (_req, res) => {
  const allowed = ['NODE_ENV', 'MIDDLE_HOST', 'MIDDLE_PORT', 'UPSTREAM_URL', 'MIDDLE_LOG_LEVEL']
  const env: Record<string, string | undefined> = {}
  for (const key of allowed) env[key] = process.env[key]
  res.json({ env, count: Object.keys(process.env).length })
})

router.get('/dependencies', (_req, res) => {
  try {
    const pkgPath = new URL('../../package.json', import.meta.url).pathname
    if (existsSync(pkgPath)) {
      const pkg = JSON.parse(readFileSync(pkgPath, 'utf-8'))
      res.json({
        dependencies: Object.keys(pkg.dependencies || {}).length,
        devDependencies: Object.keys(pkg.devDependencies || {}).length,
        total: Object.keys(pkg.dependencies || {}).length + Object.keys(pkg.devDependencies || {}).length,
      })
      return
    }
  } catch { /* ignore */ }
  res.json({ dependencies: 0, devDependencies: 0, total: 0 })
})

export default router
