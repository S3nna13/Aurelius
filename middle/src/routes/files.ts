import { Router } from 'express'
import { getEngine } from '../engine.js'
import { v4 as uuidv4 } from 'uuid'
import { existsSync, mkdirSync, writeFileSync, readFileSync, readdirSync, unlinkSync } from 'fs'
import { join } from 'path'

const router = Router()
const UPLOAD_DIR = process.env.UPLOAD_DIR || './data/uploads'

if (!existsSync(UPLOAD_DIR)) {
  mkdirSync(UPLOAD_DIR, { recursive: true })
}

interface FileRecord {
  id: string
  name: string
  size: number
  mimeType: string
  uploadedAt: string
  userId?: string
}

const fileRecords = new Map<string, FileRecord>()

router.post('/upload', (req, res) => {
  const contentType = req.headers['content-type'] || ''
  const contentLength = parseInt(req.headers['content-length'] || '0', 10)

  if (contentLength > 50 * 1024 * 1024) {
    res.status(413).json({ error: 'File too large (max 50MB)' })
    return
  }

  const chunks: Buffer[] = []
  req.on('data', (chunk) => chunks.push(chunk))
  req.on('end', () => {
    const buffer = Buffer.concat(chunks)
    const id = uuidv4()
    const name = req.headers['x-file-name'] || `upload-${id}`
    const mimeType = req.headers['x-file-type'] || contentType || 'application/octet-stream'

    const filePath = join(UPLOAD_DIR, id)
    writeFileSync(filePath, buffer)

    const record: FileRecord = {
      id,
      name: String(name),
      size: buffer.length,
      mimeType: String(mimeType),
      uploadedAt: new Date().toISOString(),
      userId: req.user?.id,
    }

    fileRecords.set(id, record)
    getEngine().appendActivity('file.upload', true, `Uploaded ${record.name} (${record.size} bytes)`)

    res.json({ success: true, file: record })
  })
})

router.get('/files', (_req, res) => {
  res.json({ files: Array.from(fileRecords.values()) })
})

router.get('/files/:id', (req, res) => {
  const record = fileRecords.get(req.params.id)
  if (!record) {
    res.status(404).json({ error: 'File not found' })
    return
  }

  const filePath = join(UPLOAD_DIR, record.id)
  if (!existsSync(filePath)) {
    res.status(404).json({ error: 'File data not found' })
    return
  }

  const data = readFileSync(filePath)
  res.setHeader('Content-Type', record.mimeType)
  res.setHeader('Content-Disposition', `attachment; filename="${record.name}"`)
  res.send(data)
})

router.delete('/files/:id', (req, res) => {
  const record = fileRecords.get(req.params.id)
  if (!record) {
    res.status(404).json({ error: 'File not found' })
    return
  }

  const filePath = join(UPLOAD_DIR, record.id)
  if (existsSync(filePath)) unlinkSync(filePath)
  fileRecords.delete(req.params.id)
  getEngine().appendActivity('file.delete', true, `Deleted ${record.name}`)
  res.json({ success: true })
})

export default router
