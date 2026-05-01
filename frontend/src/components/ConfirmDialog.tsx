import { AlertTriangle, Loader2 } from 'lucide-react'
import { Modal } from './ui/Modal'

interface ConfirmDialogProps {
  open: boolean
  onClose: () => void
  onConfirm: () => void
  title: string
  message: string
  confirmLabel?: string
  cancelLabel?: string
  variant?: 'danger' | 'warning' | 'info'
  loading?: boolean
}

const variantConfig = {
  danger: { icon: AlertTriangle, color: 'text-rose-400', bg: 'bg-rose-500/10', border: 'border-rose-500/20', btn: 'bg-rose-500 hover:bg-rose-600 text-white' },
  warning: { icon: AlertTriangle, color: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/20', btn: 'bg-amber-500 hover:bg-amber-600 text-white' },
  info: { icon: AlertTriangle, color: 'text-[#4fc3f7]', bg: 'bg-[#4fc3f7]/10', border: 'border-[#4fc3f7]/20', btn: 'aurelius-btn' },
}

export function ConfirmDialog({ open, onClose, onConfirm, title, message, confirmLabel = 'Confirm', cancelLabel = 'Cancel', variant = 'danger', loading }: ConfirmDialogProps) {
  const cfg = variantConfig[variant]

  return (
    <Modal open={open} onClose={onClose} size="sm">
      <div className="space-y-4">
        <div className={`w-10 h-10 rounded-xl ${cfg.bg} ${cfg.border} border flex items-center justify-center ${cfg.color}`}>
          <cfg.icon size={20} />
        </div>
        <div>
          <h3 className="text-sm font-semibold text-[#e0e0e0]">{title}</h3>
          <p className="text-xs text-[#9e9eb0] mt-1">{message}</p>
        </div>
        <div className="flex gap-2 justify-end pt-2">
          <button onClick={onClose} disabled={loading} className="aurelius-btn-outline text-sm disabled:opacity-50">{cancelLabel}</button>
          <button onClick={onConfirm} disabled={loading} className={`px-4 py-2 rounded-lg text-sm font-medium transition-colors disabled:opacity-50 ${cfg.btn}`}>
            {loading ? <Loader2 size={14} className="animate-spin" /> : confirmLabel}
          </button>
        </div>
      </div>
    </Modal>
  )
}
