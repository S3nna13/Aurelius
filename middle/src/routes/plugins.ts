import { Router } from 'express'

const router = Router()

const PLUGINS = [
  { id: 'code-analyzer', name: 'Code Analyzer', active: true, version: '1.0.0' },
  { id: 'data-visualizer', name: 'Data Visualizer', active: true, version: '0.5.0' },
  { id: 'security-scanner', name: 'Security Scanner', active: false, version: '2.1.0' },
  { id: 'export-tool', name: 'Export Tool', active: true, version: '1.2.0' },
]

router.get('/', (_req, res) => {
  res.json({ plugins: PLUGINS })
})

export default router
