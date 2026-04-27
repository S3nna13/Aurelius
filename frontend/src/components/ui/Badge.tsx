import type { ReactNode } from 'react'

type BadgeVariant = 'default' | 'success' | 'warning' | 'error' | 'info'
type BadgeSize = 'xs' | 'sm'

interface BadgeProps {
  children: ReactNode
  variant?: BadgeVariant
  size?: BadgeSize
  pulse?: boolean
}

const variantStyles: Record<BadgeVariant, string> = {
  default: 'bg-[#2d2d44]/30 text-[#9e9eb0] border-[#2d2d44]/50',
  success: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20',
  warning: 'bg-amber-500/10 text-amber-400 border-amber-500/20',
  error: 'bg-rose-500/10 text-rose-400 border-rose-500/20',
  info: 'bg-[#4fc3f7]/10 text-[#4fc3f7] border-[#4fc3f7]/20',
}

const sizeStyles: Record<BadgeSize, string> = {
  xs: 'text-[10px] px-1.5 py-0.5',
  sm: 'text-xs px-2 py-0.5',
}

export function Badge({ children, variant = 'default', size = 'xs', pulse = false }: BadgeProps) {
  return (
    <span className={`inline-flex items-center font-bold rounded-full border ${variantStyles[variant]} ${sizeStyles[size]} ${pulse ? 'animate-pulse' : ''}`}>
      {children}
    </span>
  )
}
