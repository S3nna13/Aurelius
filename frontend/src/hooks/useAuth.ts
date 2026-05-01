import { useState, useCallback, useEffect, useRef } from 'react'
import { useApiStore } from '../stores/apiStore'

interface AuthState {
  authenticated: boolean
  loading: boolean
  error: string | null
  login: (key: string) => Promise<boolean>
  logout: () => void
}

export function useAuth(): AuthState {
  const { apiKey, setApiKey } = useApiStore()
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const authenticated = !!apiKey

  const login = useCallback(async (key: string): Promise<boolean> => {
    setLoading(true)
    setError(null)
    try {
      const res = await fetch('/api/health', {
        headers: { 'X-API-Key': key },
      })
      if (res.ok) {
        setApiKey(key)
        return true
      }
      setError('Invalid API key')
      return false
    } catch {
      setError('Network error during authentication')
      return false
    } finally {
      setLoading(false)
    }
  }, [setApiKey])

  const logout = useCallback(() => {
    setApiKey('')
    setError(null)
  }, [setApiKey])

  const hydrated = useRef(false)
  useEffect(() => {
    if (hydrated.current) return
    hydrated.current = true
    const saved = localStorage.getItem('aurelius-api-key')
    if (saved) {
      setApiKey(saved)
    }
  }, [setApiKey])

  return { authenticated, loading, error, login, logout }
}
