export interface RedisConfig { url: string; poolSize?: number; timeoutMs?: number; retryCount?: number }
export interface RedisInfo { connected: boolean; latencyMs: number; keysCount?: number; serverVersion?: string; memoryUsed?: number }
export interface RedisHashEntry { field: string; value: string }

export declare class RedisClient {
  constructor(config: RedisConfig)
  ping(): Promise<string>
  set(key: string, value: string, ttlSeconds?: number): Promise<void>
  get(key: string): Promise<string | null>
  del(key: string): Promise<boolean>
  exists(key: string): Promise<boolean>
  expire(key: string, seconds: number): Promise<boolean>
  ttl(key: string): Promise<number>
  incr(key: string): Promise<number>
  hset(key: string, field: string, value: string): Promise<boolean>
  hget(key: string, field: string): Promise<string | null>
  hgetall(key: string): Promise<RedisHashEntry[]>
  lpush(key: string, value: string): Promise<number>
  rpop(key: string): Promise<string | null>
  lrange(key: string, start: number, stop: number): Promise<string[]>
  publish(channel: string, message: string): Promise<number>
  info(): Promise<RedisInfo>
  isConnected(): boolean
}
