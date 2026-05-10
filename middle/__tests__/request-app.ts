import { EventEmitter } from 'events'
import { PassThrough, Readable } from 'stream'
import type { Express } from 'express'

type HeaderValue = string | string[]

export interface RequestOptions {
  method?: string
  path: string
  headers?: Record<string, string>
  body?: unknown
}

export interface MockResponse {
  status: number
  headers: Record<string, HeaderValue>
  text: string
  json<T = unknown>(): Promise<T>
}

class MockRequest extends Readable {
  public readonly headers: Record<string, string>
  public readonly method: string
  public readonly url: string
  public readonly originalUrl: string
  public readonly path: string
  public readonly query: Record<string, string>
  public body: unknown
  public params: Record<string, string> = {}
  public baseUrl = ''
  public app: Express | undefined
  public res: MockResponseImpl | undefined
  public socket = Object.assign(new PassThrough(), {
    remoteAddress: '127.0.0.1',
  }) as PassThrough & { remoteAddress: string | undefined }
  public connection = this.socket
  public ip = '127.0.0.1'
  public ips: string[] = []
  public protocol = 'http'
  public secure = false
  private readonly payload: Buffer
  private pushed = false

  constructor(options: RequestOptions) {
    super()
    const method = (options.method ?? 'GET').toUpperCase()
    const parsedUrl = new URL(options.path, 'http://127.0.0.1')
    const headers = Object.fromEntries(
      Object.entries(options.headers ?? {}).map(([key, value]) => [key.toLowerCase(), value]),
    )

    let payload: Buffer
    if (options.body === undefined || options.body === null || method === 'GET' || method === 'HEAD') {
      payload = Buffer.alloc(0)
    } else if (Buffer.isBuffer(options.body)) {
      payload = options.body
    } else if (typeof options.body === 'string') {
      payload = Buffer.from(options.body)
    } else {
      payload = Buffer.from(JSON.stringify(options.body))
      if (!headers['content-type']) {
        headers['content-type'] = 'application/json'
      }
    }

    if (payload.length > 0 && !headers['content-length']) {
      headers['content-length'] = String(payload.length)
    }

    this.method = method
    this.url = `${parsedUrl.pathname}${parsedUrl.search}`
    this.originalUrl = this.url
    this.path = parsedUrl.pathname
    this.query = Object.fromEntries(parsedUrl.searchParams.entries())
    this.headers = headers
    this.body = undefined
    this.payload = payload

    this._read = this._read.bind(this)
    this.get = this.get.bind(this)
    this.header = this.header.bind(this)
  }

  override _read(): void {
    if (this.pushed) {
      return
    }

    this.pushed = true
    this.push(this.payload.length > 0 ? this.payload : null)
    if (this.payload.length === 0) {
      this.push(null)
    } else {
      this.push(null)
    }
  }

  get(name: string): string | undefined {
    const value = this.headers[name.toLowerCase()]
    return Array.isArray(value) ? value[0] : value
  }

  header(name: string): string | undefined {
    return this.get(name)
  }
}

class MockResponseImpl extends EventEmitter {
  public statusCode = 200
  public statusMessage = 'OK'
  public headersSent = false
  public writableEnded = false
  public finished = false
  public readonly locals: Record<string, unknown> = {}
  public req: MockRequest | undefined
  public app: Express | undefined
  private readonly headerStore = new Map<string, HeaderValue>()
  private readonly chunks: Buffer[] = []
  private readonly onFinish?: () => void
  private readonly onError?: (error: Error) => void
  private settled = false

  constructor(onFinish?: () => void, onError?: (error: Error) => void) {
    super()
    this.onFinish = onFinish
    this.onError = onError
    this.setHeader = this.setHeader.bind(this)
    this.getHeader = this.getHeader.bind(this)
    this.getHeaders = this.getHeaders.bind(this)
    this.getHeaderNames = this.getHeaderNames.bind(this)
    this.hasHeader = this.hasHeader.bind(this)
    this.removeHeader = this.removeHeader.bind(this)
    this.writeHead = this.writeHead.bind(this)
    this.status = this.status.bind(this)
    this.set = this.set.bind(this)
    this.type = this.type.bind(this)
    this.append = this.append.bind(this)
    this.json = this.json.bind(this)
    this.send = this.send.bind(this)
    this.write = this.write.bind(this)
    this.flushHeaders = this.flushHeaders.bind(this)
    this.redirect = this.redirect.bind(this)
    this.sendStatus = this.sendStatus.bind(this)
    this.end = this.end.bind(this)
    this.destroy = this.destroy.bind(this)
    this.text = this.text.bind(this)
  }

  setHeader(name: string, value: HeaderValue | number): this {
    this.headerStore.set(name.toLowerCase(), typeof value === 'number' ? String(value) : value)
    return this
  }

  getHeader(name: string): HeaderValue | undefined {
    return this.headerStore.get(name.toLowerCase())
  }

  getHeaders(): Record<string, HeaderValue> {
    return Object.fromEntries(this.headerStore.entries())
  }

  getHeaderNames(): string[] {
    return Array.from(this.headerStore.keys())
  }

  hasHeader(name: string): boolean {
    return this.headerStore.has(name.toLowerCase())
  }

  removeHeader(name: string): this {
    this.headerStore.delete(name.toLowerCase())
    return this
  }

  writeHead(statusCode: number, statusMessageOrHeaders?: string | Record<string, HeaderValue | number>, headers?: Record<string, HeaderValue | number>): this {
    this.statusCode = statusCode
    if (typeof statusMessageOrHeaders === 'string') {
      this.statusMessage = statusMessageOrHeaders
      if (headers) {
        for (const [key, value] of Object.entries(headers)) {
          this.setHeader(key, value)
        }
      }
    } else if (statusMessageOrHeaders) {
      for (const [key, value] of Object.entries(statusMessageOrHeaders)) {
        this.setHeader(key, value)
      }
    }
    this.headersSent = true
    return this
  }

  status(code: number): this {
    this.statusCode = code
    return this
  }

  set(field: string | Record<string, HeaderValue | number>, value?: HeaderValue | number): this {
    if (typeof field === 'string') {
      if (value !== undefined) {
        this.setHeader(field, value)
      }
      return this
    }

    for (const [key, entry] of Object.entries(field)) {
      this.setHeader(key, entry)
    }

    return this
  }

  type(value: string): this {
    this.setHeader('content-type', value)
    return this
  }

  append(field: string, value: string | string[]): this {
    const current = this.getHeader(field)
    const next = Array.isArray(value) ? value : [value]
    if (current === undefined) {
      this.setHeader(field, next.length === 1 ? next[0] : next)
      return this
    }

    const merged = Array.isArray(current) ? current.concat(next) : [current].concat(next)
    this.setHeader(field, merged.length === 1 ? merged[0] : merged)
    return this
  }

  json(payload: unknown): this {
    if (!this.hasHeader('content-type')) {
      this.setHeader('content-type', 'application/json; charset=utf-8')
    }
    return this.send(JSON.stringify(payload))
  }

  send(payload?: unknown): this {
    if (payload === undefined || payload === null) {
      return this.end()
    }

    if (Buffer.isBuffer(payload)) {
      return this.end(payload)
    }

    if (typeof payload === 'object') {
      if (!this.hasHeader('content-type')) {
        this.setHeader('content-type', 'application/json; charset=utf-8')
      }
      return this.end(JSON.stringify(payload))
    }

    return this.end(String(payload))
  }

  write(chunk: string | Buffer, encoding?: BufferEncoding): boolean {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(chunk, encoding)
    this.chunks.push(buffer)
    this.headersSent = true
    return true
  }

  flushHeaders(): void {
    this.headersSent = true
  }

  redirect(statusOrUrl: number | string, url?: string): this {
    if (typeof statusOrUrl === 'number') {
      this.status(statusOrUrl)
      if (url) {
        this.setHeader('location', url)
      }
    } else {
      this.status(302)
      this.setHeader('location', statusOrUrl)
    }
    return this.end()
  }

  sendStatus(code: number): this {
    this.status(code)
    return this.send(String(code))
  }

  end(chunk?: string | Buffer, encoding?: BufferEncoding): this {
    if (chunk !== undefined) {
      this.write(chunk, encoding)
    }
    this.headersSent = true
    this.writableEnded = true
    this.finished = true
    if (!this.settled) {
      this.settled = true
      this.onFinish?.()
    }
    queueMicrotask(() => this.emit('finish'))
    return this
  }

  destroy(error?: Error): this {
    if (error) {
      this.emit('error', error)
      this.onError?.(error)
    }
    this.writableEnded = true
    this.finished = true
    if (!this.settled) {
      this.settled = true
      this.onFinish?.()
    }
    return this
  }

  text(): string {
    return Buffer.concat(this.chunks).toString('utf8')
  }
}

export async function invokeApp(app: Express, options: RequestOptions): Promise<MockResponse> {
  const req = new MockRequest(options)
  let resolveResponse: () => void
  let rejectResponse: (error: Error) => void
  const responseDone = new Promise<void>((resolve, reject) => {
    resolveResponse = resolve
    rejectResponse = reject
  })
  const res = new MockResponseImpl(
    () => resolveResponse(),
    (error) => rejectResponse(error),
  )

  req.app = app
  req.res = res
  res.req = req
  res.app = app

  app.handle(req as never, res as never, (error?: unknown) => {
    if (error) {
      rejectResponse(error instanceof Error ? error : new Error(String(error)))
    }
  })

  await responseDone

  return {
    status: res.statusCode,
    headers: res.getHeaders(),
    text: res.text(),
    async json<T = unknown>() {
      return JSON.parse(res.text()) as T
    },
  }
}
