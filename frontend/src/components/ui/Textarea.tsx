import { forwardRef, type TextareaHTMLAttributes } from 'react'
import { AlertTriangle } from 'lucide-react'

interface TextareaProps extends TextareaHTMLAttributes<HTMLTextAreaElement> {
  label?: string
  error?: string
  hint?: string
  showCount?: boolean
  maxLength?: number
}

export const Textarea = forwardRef<HTMLTextAreaElement, TextareaProps>(
  ({ label, error, hint, showCount, maxLength, className = '', value, ...props }, ref) => {
    const charCount = typeof value === 'string' ? value.length : 0

    return (
      <div className="space-y-1.5">
        {label && (
          <label className="block text-xs text-[#9e9eb0] font-medium">
            {label}
            {props.required && <span className="text-rose-400 ml-0.5">*</span>}
          </label>
        )}
        <textarea
          ref={ref}
          value={value}
          className={`w-full bg-[#0f0f1a] border rounded-lg px-3 py-2 text-sm text-[#e0e0e0]
            placeholder:text-[#9e9eb0] focus:outline-none focus:border-[#4fc3f7]
            disabled:opacity-50 disabled:cursor-not-allowed resize-vertical transition-colors
            min-h-[80px] ${error ? 'border-rose-500' : 'border-[#2d2d44]'}
            ${className}`}
          maxLength={maxLength}
          {...props}
        />
        <div className="flex items-center justify-between">
          <div>
            {error && <p className="flex items-center gap-1 text-xs text-rose-400"><AlertTriangle size={10} />{error}</p>}
            {hint && !error && <p className="text-xs text-[#9e9eb0]/60">{hint}</p>}
          </div>
          {showCount && maxLength && (
            <span className={`text-[10px] ${charCount > maxLength * 0.9 ? 'text-rose-400' : 'text-[#9e9eb0]/40'}`}>
              {charCount}/{maxLength}
            </span>
          )}
        </div>
      </div>
    )
  },
)

Textarea.displayName = 'Textarea'

export default Textarea
