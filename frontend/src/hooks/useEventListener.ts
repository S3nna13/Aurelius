import { useEffect, useRef, type RefObject } from 'react'

type EventTarget = Window | Document | HTMLElement | RefObject<HTMLElement | null>

export function useEventListener<K extends keyof WindowEventMap>(
  event: K,
  handler: (event: WindowEventMap[K]) => void,
  target?: EventTarget,
  options?: AddEventListenerOptions,
): void

export function useEventListener<K extends keyof DocumentEventMap>(
  event: K,
  handler: (event: DocumentEventMap[K]) => void,
  target: Document,
  options?: AddEventListenerOptions,
): void

export function useEventListener<K extends keyof HTMLElementEventMap>(
  event: K,
  handler: (event: HTMLElementEventMap[K]) => void,
  target: HTMLElement,
  options?: AddEventListenerOptions,
): void

export function useEventListener(
  event: string,
  handler: (event: Event) => void,
  target?: EventTarget,
  options?: AddEventListenerOptions,
): void {
  const savedHandler = useRef(handler)

  useEffect(() => {
    savedHandler.current = handler
  }, [handler])

  useEffect(() => {
    const el = target
      ? 'current' in target
        ? target.current
        : target
      : typeof window !== 'undefined'
        ? window
        : null

    if (!el || !('addEventListener' in el)) return

    const listener = (e: Event) => savedHandler.current(e)
    el.addEventListener(event, listener, options)
    return () => el.removeEventListener(event, listener, options)
  }, [event, target, options])
}
