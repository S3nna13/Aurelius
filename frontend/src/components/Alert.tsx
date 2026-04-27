import { type ReactNode, useState } from 'react'
import { AlertTriangle, Info, CheckCircle, XCircle, X } from 'lucide-react'

type AlertVariant = 'info' | 'success' | 'warning' | 'error'

interface AlertProps {
  variant?: AlertVariant
  title?: string
  children: ReactNode
  dismissible?: boolean
  icon?: ReactNode
}

const config: Record<AlertVariant, { icon: typeof Info; classes: string; iconClasses: string }> = {
  info: { icon: Info, classes: 'bg-[#4fc3f7]/5 border-[#4fc3f7]/20 text-[#4fc3f7]', iconClasses: 'text-[#4fc3f7]' },
  success: { icon: CheckCircle, classes: 'bg-emerald-500/5 border-emerald-500/20 text-emerald-400', iconClasses: 'text-emerald-400' },
  warning: { icon: AlertTriangle, classes: 'bg-amber-500/5 border-amber-500/20 text-amber-400', iconClasses: 'text-amber-400' },
  error: { icon: XCircle, classes: 'bg-rose-500/5 border-rose-500/20 text-rose-400', iconClasses: 'text-rose-400' },
}

export function Alert({ variant = 'info', title, children, dismissible, icon }: AlertProps) {
  const [dismissed, setDismissed] = useState(false)
  if (dismissed) return null

  const { icon: DefaultIcon, classes, iconClasses } = config[variant]

  return (
    <div className={`flex items-start gap-3 px-4 py-3 rounded-lg border ${classes}`}>
      <div className={`mt-0.5 ${iconClasses}`}>{icon || <DefaultIcon size={16} />}</div>
      <div className="flex-1 min-w-0">
        {title && <p className="text-sm font-medium mb-0.5">{title}</p>}
        <div className="text-xs opacity-90">{children}</div>
      </div>
      {dismissible && (
        <button onClick={() => setDismissed(true)} className="p-0.5 opacity-60 hover:opacity-100 transition-opacity">
          <X size={14} />
        </button>
      )}
    </div>
  )
}
