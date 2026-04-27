interface ProgressProps {
  value: number
  max?: number
  label?: string
  size?: 'sm' | 'md' | 'lg'
  color?: string
  showPercent?: boolean
  animated?: boolean
}

const sizeStyles = { sm: 'h-1', md: 'h-1.5', lg: 'h-2.5' }

export default function Progress({ value, max = 100, label, size = 'md', color, showPercent = true, animated = false }: ProgressProps) {
  const pct = Math.min(100, Math.max(0, (value / max) * 100))
  const bgColor = color || (pct >= 80 ? '#34d399' : pct >= 50 ? '#fbbf24' : pct >= 25 ? '#fb923c' : '#f87171')

  return (
    <div className="w-full">
      {(label || showPercent) && (
        <div className="flex items-center justify-between mb-1">
          {label && <span className="text-xs text-[#9e9eb0]">{label}</span>}
          {showPercent && <span className="text-xs text-[#9e9eb0]">{Math.round(pct)}%</span>}
        </div>
      )}
      <div className={`w-full bg-[#0f0f1a] rounded-full overflow-hidden ${sizeStyles[size]}`}>
        <div
          className={`h-full rounded-full transition-all duration-500 ${animated ? 'animate-pulse' : ''}`}
          style={{ width: `${pct}%`, backgroundColor: bgColor }}
        />
      </div>
    </div>
  )
}
