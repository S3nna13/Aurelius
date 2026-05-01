import { useEffect } from 'react'

type KeyHandler = (e: KeyboardEvent) => void

interface KeyBinding {
  key: string
  ctrl?: boolean
  meta?: boolean
  shift?: boolean
  alt?: boolean
  handler: KeyHandler
}

export function useKeyboard(bindings: KeyBinding[], deps: unknown[] = []): void {
  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      for (const binding of bindings) {
        const ctrl = binding.ctrl ?? false
        const meta = binding.meta ?? false
        const shift = binding.shift ?? false
        const alt = binding.alt ?? false

        if (e.key.toLowerCase() !== binding.key.toLowerCase()) continue
        if (e.ctrlKey !== ctrl) continue
        if (e.metaKey !== meta) continue
        if (e.shiftKey !== shift) continue
        if (e.altKey !== alt) continue

        const target = e.target as HTMLElement
        if (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable) continue

        e.preventDefault()
        binding.handler(e)
        return
      }
    }

    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [bindings, ...deps])
}

export function useKeyboardShortcut(key: string, handler: KeyHandler, deps: unknown[] = []): void {
  useKeyboard([{ key, handler }], deps)
}

export function useKeyboardShortcutWithMod(key: string, mod: 'ctrl' | 'meta' | 'shift' | 'alt', handler: KeyHandler, deps: unknown[] = []): void {
  const modProp = { [mod]: true } as Partial<KeyBinding>
  useKeyboard([{ key, ...modProp, handler }], deps)
}
