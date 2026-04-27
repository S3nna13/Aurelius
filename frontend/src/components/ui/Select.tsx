import { forwardRef, type SelectHTMLAttributes } from 'react'
import { AlertTriangle, ChevronDown } from 'lucide-react'

interface SelectOption {
  value: string
  label: string
}

interface SelectProps extends SelectHTMLAttributes<HTMLSelectElement> {
  label?: string
  error?: string
  options: SelectOption[]
  placeholder?: string
}

export const Select = forwardRef<HTMLSelectElement, SelectProps>(
  ({ label, error, options, placeholder, className = '', ...props }, ref) => {
    return (
      <div className="space-y-1.5">
        {label && (
          <label className="block text-xs text-[#9e9eb0] font-medium">
            {label}
            {props.required && <span className="text-rose-400 ml-0.5">*</span>}
          </label>
        )}
        <div className="relative">
          <select
            ref={ref}
            className={`w-full appearance-none bg-[#0f0f1a] border rounded-lg px-3 py-2 pr-8 text-sm
              text-[#e0e0e0] focus:outline-none focus:border-[#4fc3f7]
              disabled:opacity-50 disabled:cursor-not-allowed transition-colors
              ${error ? 'border-rose-500' : 'border-[#2d2d44]'}
              ${className}`}
            {...props}
          >
            {placeholder && <option value="">{placeholder}</option>}
            {options.map((opt) => (
              <option key={opt.value} value={opt.value}>{opt.label}</option>
            ))}
          </select>
          <ChevronDown size={14} className="absolute right-3 top-1/2 -translate-y-1/2 text-[#9e9eb0] pointer-events-none" />
        </div>
        {error && <p className="flex items-center gap-1 text-xs text-rose-400"><AlertTriangle size={10} />{error}</p>}
      </div>
    )
  },
)

Select.displayName = 'Select'
