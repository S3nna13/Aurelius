import { createClient, type RedisClientType } from 'redis'
import { config } from './config.js'

let _client: RedisClientType | null = null

export async function getCache(): Promise<RedisClientType | null> {
  if (_client?.isOpen) return _client
  try {
    _client = createClient({ url: config.redisUrl })
    _client.on('error', (err) => console.warn('[cache] redis error:', err.message))
    await _client.connect()
    return _client
  } catch (err) {
    console.warn('[cache] connection failed:', err instanceof Error ? err.message : err)
    return null
  }
}

export async function cacheGet<T>(key: string): Promise<T | null> {
  const client = await getCache()
  if (!client) return null
  try {
    const raw = await client.get(key)
    return raw ? JSON.parse(raw) as T : null
  } catch (err) {
    console.warn('[cache] get failed:', err instanceof Error ? err.message : err)
    return null
  }
}

export async function cacheSet(key: string, value: unknown, ttl = 60): Promise<void> {
  const client = await getCache()
  if (!client) return
  try {
    await client.setEx(key, ttl, JSON.stringify(value))
  } catch (err) {
    console.warn('[cache] set failed:', err instanceof Error ? err.message : err)
  }
}

export async function cacheDel(key: string): Promise<void> {
  const client = await getCache()
  if (!client) return
  try {
    await client.del(key)
  } catch (err) {
    console.warn('[cache] del failed:', err instanceof Error ? err.message : err)
  }
}

export async function cacheIncr(key: string): Promise<number> {
  const client = await getCache()
  if (!client) return 0
  try {
    return await client.incr(key)
  } catch (err) {
    console.warn('[cache] incr failed:', err instanceof Error ? err.message : err)
    return 0
  }
}

export async function cacheExpire(key: string, ttl: number): Promise<void> {
  const client = await getCache()
  if (!client) return
  try {
    await client.expire(key, ttl)
  } catch (err) {
    console.warn('[cache] expire failed:', err instanceof Error ? err.message : err)
  }
}
