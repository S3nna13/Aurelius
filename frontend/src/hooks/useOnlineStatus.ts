import { useState, useEffect } from 'react'

interface UseOnlineStatusReturn {
  online: boolean
  wasOffline: boolean
  lastOnline: Date | null
  lastOffline: Date | null
}

export function useOnlineStatus(): UseOnlineStatusReturn {
  const [online, setOnline] = useState(navigator.onLine)
  const [wasOffline, setWasOffline] = useState(false)
  const [lastOnline, setLastOnline] = useState<Date | null>(navigator.onLine ? new Date() : null)
  const [lastOffline, setLastOffline] = useState<Date | null>(null)

  useEffect(() => {
    const goOnline = () => {
      setOnline(true)
      setLastOnline(new Date())
    }
    const goOffline = () => {
      setOnline(false)
      setWasOffline(true)
      setLastOffline(new Date())
    }

    window.addEventListener('online', goOnline)
    window.addEventListener('offline', goOffline)
    return () => {
      window.removeEventListener('online', goOnline)
      window.removeEventListener('offline', goOffline)
    }
  }, [])

  return { online, wasOffline, lastOnline, lastOffline }
}
