import { useEffect, useRef } from 'react'

export default function SkipToContent() {
  const ref = useRef<HTMLAnchorElement>(null)

  useEffect(() => {
    const handler = (e: KeyboardEvent) => {
      if (e.key === 'Tab' && !e.shiftKey && ref.current) {
        ref.current.focus()
      }
    }
    window.addEventListener('keydown', handler)
    return () => window.removeEventListener('keydown', handler)
  }, [])

  return (
    <a
      ref={ref}
      href="#main-content"
      className="sr-only focus:not-sr-only focus:absolute focus:top-4 focus:left-4 focus:z-[200] focus:px-4 focus:py-2 focus:bg-[#4fc3f7] focus:text-[#0f0f1a] focus:rounded-lg focus:text-sm focus:font-bold"
    >
      Skip to main content
    </a>
  )
}
