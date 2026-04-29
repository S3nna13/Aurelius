import { createElement, isValidElement, type ReactNode } from 'react'
import { Inbox } from 'lucide-react'

type IconLike = ReactNode | object

type EmptyStateAction =
  | ReactNode
  | {
      label: string
      onClick: () => void
      className?: string
    }

interface EmptyStateProps {
  icon?: IconLike
  title: string
  description?: string
  action?: EmptyStateAction
}

export function EmptyState({ icon, title, description, action }: EmptyStateProps) {
  const actionObject = action && typeof action === 'object' && 'label' in action && 'onClick' in action
    ? action as Exclude<EmptyStateAction, ReactNode>
    : null
  const iconNode = isValidElement(icon)
    ? icon
    : icon && (typeof icon === 'function' || typeof icon === 'object')
      ? createElement(icon as any, { size: 24 })
      : icon

  return (
    <div className="flex flex-col items-center justify-center py-12 px-6 text-center">
      <div className="w-12 h-12 rounded-xl bg-[#2d2d44]/30 border border-[#2d2d44]/50 flex items-center justify-center mb-4 text-[#9e9eb0]">
        {iconNode || <Inbox size={24} />}
      </div>
      <h3 className="text-sm font-medium text-[#e0e0e0]">{title}</h3>
      {description && <p className="text-xs text-[#9e9eb0] mt-1 max-w-sm">{description}</p>}
      {action && (
        <div className="mt-4">
          {actionObject ? (
            <button
              onClick={actionObject.onClick}
              className={`aurelius-btn-primary text-sm ${actionObject.className ?? ''}`}
            >
              {actionObject.label}
            </button>
          ) : (
            action as ReactNode
          )}
        </div>
      )}
    </div>
  )
}

export default EmptyState
