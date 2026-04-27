import { useEffect, useRef, type RefObject } from 'react'

export function useClickOutside<T extends HTMLElement>(handler: () => void): RefObject<T | null> {
  const ref = useRef<T | null>(null)

  useEffect(() => {
    const listener = (e: MouseEvent | TouchEvent) => {
      if (!ref.current || ref.current.contains(e.target as Node)) return
      handler()
    }
    document.addEventListener('mousedown', listener)
    document.addEventListener('touchstart', listener)
    return () => {
      document.removeEventListener('mousedown', listener)
      document.removeEventListener('touchstart', listener)
    }
  }, [handler])

  return ref
}
