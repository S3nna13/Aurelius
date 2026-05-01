import { type ReactNode } from 'react'

interface ContainerProps {
  children: ReactNode
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl' | 'full'
  className?: string
}

const widths = { sm: 'max-w-2xl', md: 'max-w-4xl', lg: 'max-w-6xl', xl: 'max-w-7xl', full: 'max-w-full' }

export function Container({ children, maxWidth = 'lg', className = '' }: ContainerProps) {
  return (
    <div className={`mx-auto px-4 sm:px-6 ${widths[maxWidth]} ${className}`}>
      {children}
    </div>
  )
}

export function Section({ children, title, subtitle, action, className = '' }: { children: ReactNode; title?: string; subtitle?: string; action?: ReactNode; className?: string }) {
  return (
    <div className={`space-y-4 ${className}`}>
      {(title || action) && (
        <div className="flex items-center justify-between gap-4">
          <div>
            {title && <h2 className="text-sm font-semibold text-[#e0e0e0] uppercase tracking-wider">{title}</h2>}
            {subtitle && <p className="text-xs text-[#9e9eb0] mt-0.5">{subtitle}</p>}
          </div>
          {action && <div>{action}</div>}
        </div>
      )}
      {children}
    </div>
  )
}

export function Grid({ children, cols = 1, gap = 4, className = '' }: { children: ReactNode; cols?: 1 | 2 | 3 | 4; gap?: number; className?: string }) {
  const gridCols = { 1: 'grid-cols-1', 2: 'grid-cols-1 sm:grid-cols-2', 3: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-3', 4: 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-4' }
  return (
    <div className={`grid ${gridCols[cols]} gap-${gap} ${className}`}>
      {children}
    </div>
  )
}
