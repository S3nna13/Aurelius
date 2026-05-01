export interface Session {
  id: string
  userId: string
  role: string
  createdAt: string
  expiresAt: string
  lastActivity: string
  metadataJson: string
  ipAddress: string
  userAgent: string
}

export interface SessionCreateOptions {
  userId: string
  role: string
  ttlSeconds?: number
  metadata?: string
  ipAddress?: string
  userAgent?: string
}

export interface SessionStats {
  totalSessions: number
  activeSessions: number
  expiredSessions: number
}

export interface SessionValidation {
  valid: boolean
  session: Session | null
  reason: string | null
}

export declare class SessionManager {
  constructor()
  createSession(options: SessionCreateOptions): Session
  getSession(sessionId: string): Session | null
  validateSession(sessionId: string): SessionValidation
  touchSession(sessionId: string): boolean
  deleteSession(sessionId: string): boolean
  listUserSessions(userId: string): Session[]
  deleteUserSessions(userId: string): number
  getStats(): SessionStats
  cleanup(): number
}
