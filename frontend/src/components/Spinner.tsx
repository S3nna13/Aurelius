import { Loader2 } from 'lucide-react'

interface SpinnerProps {
  size?: number
  text?: string
  fullPage?: boolean
}

export function Spinner({ size = 24, text, fullPage }: SpinnerProps) {
  if (fullPage) {
    return (
      <div className="flex flex-col items-center justify-center min-h-[50vh] gap-3">
        <Loader2 size={size} className="animate-spin text-[#4fc3f7]" />
        {text && <p className="text-sm text-[#9e9eb0]">{text}</p>}
      </div>
    )
  }

  return (
    <div className="flex items-center gap-2">
      <Loader2 size={size} className="animate-spin text-[#4fc3f7]" />
      {text && <span className="text-sm text-[#9e9eb0]">{text}</span>}
    </div>
  )
}
