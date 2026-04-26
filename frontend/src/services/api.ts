const DEFAULT_TIMEOUT = 15000
const DEFAULT_RETRIES = 2

interface ApiOptions {
  method?: 'GET' | 'POST' | 'PUT' | 'DELETE' | 'PATCH'
  headers?: Record<string, string>
  body?: unknown
  params?: Record<string, string | number | boolean | undefined>
  timeout?: number
  retries?: number
}

interface ApiResponse<T> {
  data: T | null
  error: string | null
  status: number
}

function buildUrl(base: string, path: string, params?: Record<string, string | number | boolean | undefined>): string {
  const url = new URL(path.startsWith('/') ? path : `/${path}`, base.endsWith('/') ? base : `${base}/`)
  if (params) {
    Object.entries(params).forEach(([k, v]) => {
      if (v !== undefined) url.searchParams.set(k, String(v))
    })
  }
  return url.toString()
}

async function request<T>(baseUrl: string, path: string, opts: ApiOptions): Promise<ApiResponse<T>> {
  const { method = 'GET', headers = {}, body, params, timeout = DEFAULT_TIMEOUT, retries = DEFAULT_RETRIES } = opts

  const controller = new AbortController()
  const timer = setTimeout(() => controller.abort(), timeout)

  const apiKey = localStorage.getItem('aurelius-api-key') || ''

  let lastErr: Error | null = null

  for (let attempt = 0; attempt <= retries; attempt++) {
    try {
      const res = await fetch(buildUrl(baseUrl, path, params), {
        method,
        headers: {
          'Content-Type': 'application/json',
          ...(apiKey ? { 'X-API-Key': apiKey } : {}),
          ...headers,
        },
        body: body ? JSON.stringify(body) : undefined,
        signal: controller.signal,
      })

      clearTimeout(timer)

      const text = await res.text()
      let data: T | null = null
      try {
        data = text ? JSON.parse(text) : null
      } catch {
        // non-JSON response
      }

      if (!res.ok) {
        const errMsg = (data as { error?: { message?: string } })?.error?.message || `HTTP ${res.status}`
        return { data: null, error: errMsg, status: res.status }
      }

      return { data, error: null, status: res.status }
    } catch (err) {
      lastErr = err instanceof Error ? err : new Error(String(err))
      if (attempt < retries && (err as Error).name !== 'AbortError') {
        await new Promise((r) => setTimeout(r, Math.min(1000 * 2 ** attempt, 5000)))
      }
    }
  }

  clearTimeout(timer)
  const message = (lastErr?.name === 'AbortError') ? 'Request timed out' : (lastErr?.message || 'Unknown error')
  return { data: null, error: message, status: 0 }
}

function apiClient(baseUrl?: string) {
  const base = baseUrl || import.meta.env.VITE_API_BASE_URL || '/api'

  return {
    get: <T>(path: string, opts?: Omit<ApiOptions, 'method' | 'body'>) =>
      request<T>(base, path, { ...opts, method: 'GET' }),

    post: <T>(path: string, body?: unknown, opts?: Omit<ApiOptions, 'method' | 'body'>) =>
      request<T>(base, path, { ...opts, method: 'POST', body }),

    put: <T>(path: string, body?: unknown, opts?: Omit<ApiOptions, 'method' | 'body'>) =>
      request<T>(base, path, { ...opts, method: 'PUT', body }),

    delete: <T>(path: string, opts?: Omit<ApiOptions, 'method' | 'body'>) =>
      request<T>(base, path, { ...opts, method: 'DELETE' }),

    patch: <T>(path: string, body?: unknown, opts?: Omit<ApiOptions, 'method' | 'body'>) =>
      request<T>(base, path, { ...opts, method: 'PATCH', body }),
  }
}

export const api = apiClient()
export { apiClient }
export type { ApiOptions, ApiResponse }
