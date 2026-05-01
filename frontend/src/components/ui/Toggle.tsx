import { type InputHTMLAttributes } from 'react'

interface ToggleProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type' | 'onChange'> {
  label?: string
  description?: string
  onChange?: (checked: boolean) => void
}

export function Toggle({ label, description, className = '', onChange, checked, ...props }: ToggleProps) {
  return (
    <label className={`flex items-start gap-3 py-2 cursor-pointer ${className}`}>
      <div className="relative mt-0.5">
        <input
          type="checkbox"
          className="sr-only peer"
          checked={checked}
          onChange={(e) => onChange?.(e.target.checked)}
          {...props}
        />
        <div className="w-9 h-5 bg-[#2d2d44] rounded-full peer-checked:bg-[#4fc3f7] transition-colors
          after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:w-4 after:h-4
          after:bg-white after:rounded-full after:transition-transform peer-checked:after:translate-x-4
          peer-disabled:opacity-50 peer-disabled:cursor-not-allowed" />
      </div>
      {(label || description) && (
        <div>
          {label && <p className="text-sm text-[#e0e0e0]">{label}</p>}
          {description && <p className="text-xs text-[#9e9eb0] mt-0.5">{description}</p>}
        </div>
      )}
    </label>
  )
}

export default Toggle
