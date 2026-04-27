import { useState, useRef, type ReactNode } from 'react'

interface TooltipProps {
  content: ReactNode
  children: ReactNode
  position?: 'top' | 'bottom' | 'left' | 'right'
  delay?: number
}

export default function Tooltip({ content, children, position = 'top', delay = 300 }: TooltipProps) {
  const [visible, setVisible] = useState(false)
  const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)

  const show = () => {
    clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => setVisible(true), delay)
  }

  const hide = () => {
    clearTimeout(timerRef.current)
    setVisible(false)
  }

  const positionStyles: Record<string, string> = {
    top: 'bottom-full left-1/2 -translate-x-1/2 mb-2',
    bottom: 'top-full left-1/2 -translate-x-1/2 mt-2',
    left: 'right-full top-1/2 -translate-y-1/2 mr-2',
    right: 'left-full top-1/2 -translate-y-1/2 ml-2',
  }

  return (
    <div className="relative inline-flex" onMouseEnter={show} onMouseLeave={hide} onFocus={show} onBlur={hide}>
      {children}
      {visible && (
        <div className={`absolute z-50 ${positionStyles[position]} pointer-events-none`}>
          <div className="bg-[#2d2d44] text-[#e0e0e0] text-xs rounded-lg px-2.5 py-1.5 shadow-lg border border-[#3d3d54] whitespace-nowrap max-w-xs">
            {content}
          </div>
        </div>
      )}
    </div>
  )
}
