import { useState, useCallback } from 'react'

interface UseClipboardReturn {
  copied: boolean
  copy: (text: string) => Promise<boolean>
  error: Error | null
}

export function useClipboard(resetDelay = 2000): UseClipboardReturn {
  const [copied, setCopied] = useState(false)
  const [error, setError] = useState<Error | null>(null)

  const copy = useCallback(async (text: string): Promise<boolean> => {
    try {
      await navigator.clipboard.writeText(text)
      setCopied(true)
      setError(null)
      setTimeout(() => setCopied(false), resetDelay)
      return true
    } catch (err) {
      const e = err instanceof Error ? err : new Error(String(err))
      setError(e)
      return false
    }
  }, [resetDelay])

  return { copied, copy, error }
}
