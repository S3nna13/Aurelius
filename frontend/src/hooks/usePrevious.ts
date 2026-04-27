import { useRef, useEffect } from 'react'

export function usePrevious<T>(value: T): T | undefined {
  const ref = useRef<T | undefined>(undefined)

  useEffect(() => {
    ref.current = value
  }, [value])

  return ref.current
}

export function usePreviousImmediate<T>(value: T): T | undefined {
  const ref = useRef<T | undefined>(undefined)

  const prev = ref.current
  ref.current = value

  return prev
}
