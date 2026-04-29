import { Router } from 'express'
import { getEngine } from '../engine.js'
import { v4 as uuidv4 } from 'uuid'
import { existsSync, mkdirSync, writeFileSync, readFileSync, unlinkSync } from 'fs'
import { join } from 'path'

const router = Router()
const UPLOAD_DIR = process.env.UPLOAD_DIR || './data/uploads'
const MAX_FILE_SIZE = 50 * 1024 * 1024
const USER_QUOTA_LIMIT = 500 * 1024 * 1024

if (!existsSync(UPLOAD_DIR)) {
  mkdirSync(UPLOAD_DIR, { recursive: true })
}

interface FileRecord {
  id: string
  name: string
  size: number
  mimeType: string
  uploadedAt: string
  userId: string
}

const fileRecords = new Map<string, FileRecord>()
const userQuota = new Map<string, number>()

router.post('/upload', (req, res) => {
  const userId = req.user?.id
  if (!userId) {
    res.status(401).json({ error: 'Unauthorized' })
    return
  }

  const contentType = req.headers['content-type'] || ''
  const contentLength = parseInt(req.headers['content-length'] || '0', 10)

  if (contentLength > MAX_FILE_SIZE) {
    res.status(413).json({ error: 'File too large (max 50MB)' })
    return
  }

  const currentQuota = userQuota.get(userId) || 0

  let uploaded = 0
  const chunks: Buffer[] = []
  let aborted = false

  req.on('data', (chunk: Buffer) => {
    uploaded += chunk.length
    if (uploaded > MAX_FILE_SIZE) {
      aborted = true
      req.destroy()
      return
    }
    chunks.push(chunk)
  })

  req.on('end', () => {
    if (aborted) {
      res.status(413).json({ error: 'File too large (max 50MB)' })
      return
    }

    const buffer = Buffer.concat(chunks)
    if (buffer.length === 0) {
      res.status(400).json({ error: 'Empty upload' })
      return
    }

    if (currentQuota + buffer.length > USER_QUOTA_LIMIT) {
      res.status(413).json({ error: 'Quota exceeded' })
      return
    }

    const id = uuidv4()
    const name = String(req.headers['x-file-name'] || `upload-${id}`).replace(/[\r\n]/g, '')
    const mimeType = req.headers['x-file-type'] || contentType || 'application/octet-stream'

    const filePath = join(UPLOAD_DIR, id)
    writeFileSync(filePath, buffer)

    const record: FileRecord = {
      id,
      name: String(name),
      size: buffer.length,
      mimeType: String(mimeType),
      uploadedAt: new Date().toISOString(),
      userId,
    }

    fileRecords.set(id, record)
    userQuota.set(userId, currentQuota + buffer.length)
    getEngine().appendActivity('file.upload', true, `Uploaded ${record.name} (${record.size} bytes)`)

    res.json({ success: true, file: record })
  })
})

router.get('/files', (_req, res) => {
  const records = Array.from(fileRecords.values())
    .filter(r => r.userId === _req.user?.id || _req.user?.role === 'admin')
  res.json({ files: records })
})

router.get('/files/:id', (req, res) => {
  const record = fileRecords.get(req.params.id)
  if (!record) {
    res.status(404).json({ error: 'File not found' })
    return
  }

  if (record.userId !== req.user?.id && req.user?.role !== 'admin') {
    res.status(403).json({ error: 'Forbidden' })
    return
  }

  const filePath = join(UPLOAD_DIR, record.id)
  if (!existsSync(filePath)) {
    res.status(404).json({ error: 'File data not found' })
    return
  }

  const data = readFileSync(filePath)
  res.setHeader('Content-Type', record.mimeType)
  const safeName = record.name.replace(/[\r\n"\\]/g, '')
  res.setHeader('Content-Disposition', `attachment; filename="${safeName}"`)
  res.send(data)
})

router.delete('/files/:id', (req, res) => {
  const record = fileRecords.get(req.params.id)
  if (!record) {
    res.status(404).json({ error: 'File not found' })
    return
  }

  if (record.userId !== req.user?.id && req.user?.role !== 'admin') {
    res.status(403).json({ error: 'Forbidden' })
    return
  }

  const filePath = join(UPLOAD_DIR, record.id)
  if (existsSync(filePath)) unlinkSync(filePath)
  fileRecords.delete(req.params.id)
  getEngine().appendActivity('file.delete', true, `Deleted ${record.name}`)
  res.json({ success: true })
})

export default router
