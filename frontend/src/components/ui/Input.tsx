import { forwardRef, type InputHTMLAttributes } from 'react'
import { AlertTriangle } from 'lucide-react'

interface InputProps extends InputHTMLAttributes<HTMLInputElement> {
  label?: string
  error?: string
  hint?: string
  icon?: React.ReactNode
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  ({ label, error, hint, icon, className = '', ...props }, ref) => {
    return (
      <div className="space-y-1.5">
        {label && (
          <label className="block text-xs text-[#9e9eb0] font-medium">
            {label}
            {props.required && <span className="text-rose-400 ml-0.5">*</span>}
          </label>
        )}
        <div className="relative">
          {icon && (
            <div className="absolute left-3 top-1/2 -translate-y-1/2 text-[#9e9eb0] pointer-events-none">
              {icon}
            </div>
          )}
          <input
            ref={ref}
            className={`w-full bg-[#0f0f1a] border rounded-lg px-3 py-2 text-sm text-[#e0e0e0]
              placeholder:text-[#9e9eb0] focus:outline-none focus:border-[#4fc3f7]
              disabled:opacity-50 disabled:cursor-not-allowed transition-colors
              ${icon ? 'pl-9' : ''}
              ${error ? 'border-rose-500 focus:border-rose-500' : 'border-[#2d2d44]'}
              ${className}`}
            {...props}
          />
        </div>
        {error && <p className="flex items-center gap-1 text-xs text-rose-400"><AlertTriangle size={10} />{error}</p>}
        {hint && !error && <p className="text-xs text-[#9e9eb0]/60">{hint}</p>}
      </div>
    )
  },
)

Input.displayName = 'Input'

export default Input
