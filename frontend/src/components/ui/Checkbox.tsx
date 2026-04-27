import { type InputHTMLAttributes } from 'react'

interface CheckboxProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string
}

export function Checkbox({ label, className = '', ...props }: CheckboxProps) {
  return (
    <label className={`flex items-center gap-2.5 py-1 cursor-pointer ${className}`}>
      <input type="checkbox" className="w-4 h-4 rounded border-[#2d2d44] bg-[#0f0f1a]
        text-[#4fc3f7] focus:ring-[#4fc3f7]/30 focus:ring-2 accent-[#4fc3f7]
        disabled:opacity-50 disabled:cursor-not-allowed transition-colors" {...props} />
      {label && <span className="text-sm text-[#e0e0e0] select-none">{label}</span>}
    </label>
  )
}
