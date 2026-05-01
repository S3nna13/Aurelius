import { useState, useCallback, useRef, useEffect } from 'react'

interface UseAsyncState<T> {
  data: T | null
  loading: boolean
  error: Error | null
}

interface UseAsyncReturn<T> extends UseAsyncState<T> {
  execute: (...args: unknown[]) => Promise<T | null>
  reset: () => void
}

export function useAsync<T>(
  asyncFn: (...args: unknown[]) => Promise<T>,
  immediate = false,
): UseAsyncReturn<T> {
  const [state, setState] = useState<UseAsyncState<T>>({ data: null, loading: immediate, error: null })
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true
    return () => { mountedRef.current = false }
  }, [])

  const execute = useCallback(async (...args: unknown[]): Promise<T | null> => {
    setState((prev) => ({ ...prev, loading: true, error: null }))
    try {
      const result = await asyncFn(...args)
      if (mountedRef.current) setState({ data: result, loading: false, error: null })
      return result
    } catch (err) {
      const error = err instanceof Error ? err : new Error(String(err))
      if (mountedRef.current) setState({ data: null, loading: false, error })
      return null
    }
  }, [asyncFn])

  const reset = useCallback(() => {
    if (mountedRef.current) setState({ data: null, loading: false, error: null })
  }, [])

  return { ...state, execute, reset }
}
