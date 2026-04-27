import { useState, useRef, useEffect, type ReactNode } from 'react'
import { ChevronDown } from 'lucide-react'

interface DropdownItem {
  label: string
  value: string
  icon?: ReactNode
  disabled?: boolean
  divider?: boolean
  danger?: boolean
}

interface DropdownProps {
  items: DropdownItem[]
  value?: string
  onChange?: (value: string) => void
  placeholder?: string
  trigger?: ReactNode
  align?: 'left' | 'right'
}

export function Dropdown({ items, value, onChange, placeholder = 'Select...', trigger, align = 'left' }: DropdownProps) {
  const [open, setOpen] = useState(false)
  const ref = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const selected = items.find((i) => i.value === value)

  return (
    <div ref={ref} className="relative">
      {trigger ? (
        <div onClick={() => setOpen(!open)}>{trigger}</div>
      ) : (
        <button onClick={() => setOpen(!open)}
          className="flex items-center justify-between gap-2 w-full bg-[#0f0f1a] border border-[#2d2d44] rounded-lg px-3 py-2 text-sm text-[#e0e0e0] hover:border-[#4fc3f7]/30 transition-colors">
          <span className={selected ? '' : 'text-[#9e9eb0]'}>{selected?.label || placeholder}</span>
          <ChevronDown size={14} className={`text-[#9e9eb0] transition-transform ${open ? 'rotate-180' : ''}`} />
        </button>
      )}

      {open && (
        <div className={`absolute z-50 mt-1 w-full min-w-[180px] bg-[#1a1a2e] border border-[#2d2d44] rounded-lg shadow-xl py-1 ${align === 'right' ? 'right-0' : 'left-0'}`}>
          {items.map((item, i) => (
            <div key={item.value}>
              {item.divider && i > 0 && <div className="mx-2 my-1 border-t border-[#2d2d44]" />}
              <button onClick={() => { if (!item.disabled) { onChange?.(item.value); setOpen(false) } }}
                disabled={item.disabled}
                className={`w-full flex items-center gap-2 px-3 py-2 text-sm transition-colors ${
                  item.disabled ? 'opacity-40 cursor-not-allowed' : ''
                } ${
                  value === item.value ? 'bg-[#4fc3f7]/10 text-[#4fc3f7]' : 'text-[#e0e0e0] hover:bg-[#2d2d44]/40'
                } ${item.danger ? 'text-rose-400 hover:bg-rose-500/10' : ''}`}>
                {item.icon && <span className="w-4 h-4">{item.icon}</span>}
                {item.label}
              </button>
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
