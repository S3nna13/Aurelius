import { useRef, useState, useCallback } from 'react'

interface Rect {
  left: number
  top: number
  width: number
  height: number
  bottom: number
  right: number
  x: number
  y: number
}

export function useMeasure<T extends HTMLElement>(): [
  React.RefObject<T | null>,
  Rect,
  () => void,
] {
  const ref = useRef<T | null>(null)
  const [rect, setRect] = useState<Rect>({
    left: 0, top: 0, width: 0, height: 0,
    bottom: 0, right: 0, x: 0, y: 0,
  })

  const measure = useCallback(() => {
    if (ref.current) {
      const r = ref.current.getBoundingClientRect()
      setRect({
        left: r.left, top: r.top, width: r.width, height: r.height,
        bottom: r.bottom, right: r.right, x: r.x, y: r.y,
      })
    }
  }, [])

  return [ref, rect, measure]
}
