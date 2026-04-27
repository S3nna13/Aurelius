import { type ReactNode } from 'react'
import { Inbox } from 'lucide-react'

interface EmptyStateProps {
  icon?: ReactNode
  title: string
  description?: string
  action?: ReactNode
}

export function EmptyState({ icon, title, description, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center justify-center py-12 px-6 text-center">
      <div className="w-12 h-12 rounded-xl bg-[#2d2d44]/30 border border-[#2d2d44]/50 flex items-center justify-center mb-4 text-[#9e9eb0]">
        {icon || <Inbox size={24} />}
      </div>
      <h3 className="text-sm font-medium text-[#e0e0e0]">{title}</h3>
      {description && <p className="text-xs text-[#9e9eb0] mt-1 max-w-sm">{description}</p>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  )
}
