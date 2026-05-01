interface SkeletonProps {
  className?: string
  count?: number
  height?: string
  width?: string
  rounded?: string
}

export function Skeleton({ className = '', count = 1, height = 'h-4', width = 'w-full', rounded = 'rounded' }: SkeletonProps) {
  return (
    <>
      {Array.from({ length: count }).map((_, i) => (
        <div
          key={i}
          className={`${height} ${width} ${rounded} bg-[#2d2d44]/50 animate-pulse ${className}`}
        />
      ))}
    </>
  )
}

export function SkeletonCard() {
  return (
    <div className="aurelius-card space-y-3">
      <div className="flex justify-between">
        <Skeleton width="w-8" height="h-8" rounded="rounded-lg" />
        <Skeleton width="w-12" height="h-4" rounded="rounded-full" />
      </div>
      <Skeleton width="w-20" height="h-7" />
      <Skeleton width="w-24" height="h-3" />
    </div>
  )
}

export function SkeletonTable({ rows = 5, cols = 4 }: { rows?: number; cols?: number }) {
  return (
    <div className="space-y-2">
      <div className="flex gap-4 pb-3 border-b border-[#2d2d44]">
        {Array.from({ length: cols }).map((_, i) => (
          <Skeleton key={i} width="flex-1" height="h-3" />
        ))}
      </div>
      {Array.from({ length: rows }).map((_, r) => (
        <div key={r} className="flex gap-4 py-3 border-b border-[#2d2d44]/30">
          {Array.from({ length: cols }).map((_, c) => (
            <Skeleton key={c} width="flex-1" height="h-4" />
          ))}
        </div>
      ))}
    </div>
  )
}

export default Skeleton
