import { type ReactNode } from 'react'
import { ArrowUp, ArrowDown, Minus } from 'lucide-react'

interface StatsCardProps {
  title: string
  value: string | number
  icon?: ReactNode
  change?: number
  changeLabel?: string
  color?: string
  subtitle?: string
  loading?: boolean
  onClick?: () => void
}

export function StatsCard({ title, value, icon, change, changeLabel, color = '#4fc3f7', subtitle, loading, onClick }: StatsCardProps) {
  const changeIcon = change !== undefined
    ? change > 0 ? <ArrowUp size={12} /> : change < 0 ? <ArrowDown size={12} /> : <Minus size={12} />
    : null

  const changeColor = change !== undefined
    ? change > 0 ? 'text-emerald-400' : change < 0 ? 'text-rose-400' : 'text-[#9e9eb0]'
    : ''

  return (
    <div
      className={`aurelius-card space-y-3 ${onClick ? 'cursor-pointer hover:border-[#4fc3f7]/30 transition-colors' : ''}`}
      onClick={onClick}
    >
      <div className="flex items-center justify-between">
        {icon && <div className="text-[color]" style={{ color }}>{icon}</div>}
        {loading && <div className="w-16 h-4 bg-[#2d2d44] rounded animate-pulse" />}
      </div>
      {loading ? (
        <div className="space-y-2">
          <div className="w-20 h-7 bg-[#2d2d44] rounded animate-pulse" />
          <div className="w-24 h-3 bg-[#2d2d44] rounded animate-pulse" />
        </div>
      ) : (
        <>
          <p className="text-2xl font-bold" style={{ color }}>{value}</p>
          <div className="flex items-center gap-2">
            <p className="text-xs text-[#9e9eb0] uppercase tracking-wider">{title}</p>
            {(change !== undefined || subtitle) && (
              <div className="flex items-center gap-1">
                {changeIcon && change !== undefined && <span className={`flex items-center gap-0.5 text-[10px] font-bold ${changeColor}`}>{changeIcon}{Math.abs(change)}%</span>}
                {changeLabel && <span className="text-[10px] text-[#9e9eb0]/60">{changeLabel}</span>}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  )
}
