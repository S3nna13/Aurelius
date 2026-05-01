import { useState, useEffect, useCallback } from 'react'

interface NetworkInfo {
  downlink: number
  effectiveType: string
  rtt: number
  saveData: boolean
}

export function useNetworkStatus() {
  const [info, setInfo] = useState<NetworkInfo>(() => {
    if ('connection' in navigator) {
      const c = (navigator as any).connection
      return { downlink: c.downlink, effectiveType: c.effectiveType, rtt: c.rtt, saveData: c.saveData }
    }
    return { downlink: 0, effectiveType: 'unknown', rtt: 0, saveData: false }
  })

  const update = useCallback(() => {
    if ('connection' in navigator) {
      const c = (navigator as any).connection
      setInfo({ downlink: c.downlink, effectiveType: c.effectiveType, rtt: c.rtt, saveData: c.saveData })
    }
  }, [])

  useEffect(() => {
    if ('connection' in navigator) {
      const c = (navigator as any).connection
      c.addEventListener('change', update)
      return () => c.removeEventListener('change', update)
    }
  }, [update])

  return info
}
