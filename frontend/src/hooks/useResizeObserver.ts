import { useEffect, useRef, useState, type RefObject } from 'react'

interface Size {
  width: number
  height: number
}

export function useResizeObserver<T extends HTMLElement>(): [RefObject<T | null>, Size] {
  const ref = useRef<T | null>(null)
  const [size, setSize] = useState<Size>({ width: 0, height: 0 })

  useEffect(() => {
    const el = ref.current
    if (!el) return

    const observer = new ResizeObserver((entries) => {
      for (const entry of entries) {
        const { width, height } = entry.contentRect
        setSize({ width: Math.round(width), height: Math.round(height) })
      }
    })

    observer.observe(el)
    return () => observer.disconnect()
  }, [])

  return [ref, size]
}
