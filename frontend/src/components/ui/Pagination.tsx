import { ChevronLeft, ChevronRight } from 'lucide-react'

interface PaginationProps {
  page: number
  totalPages: number
  onPageChange: (page: number) => void
  total?: number
}

export function Pagination({ page, totalPages, onPageChange, total }: PaginationProps) {
  if (totalPages <= 1) return null

  const pages: (number | string)[] = []
  for (let i = 1; i <= totalPages; i++) {
    if (i === 1 || i === totalPages || (i >= page - 1 && i <= page + 1)) {
      pages.push(i)
    } else if (pages[pages.length - 1] !== '...') {
      pages.push('...')
    }
  }

  return (
    <div className="flex items-center justify-between pt-4">
      {total !== undefined && <p className="text-xs text-[#9e9eb0]">{total} total</p>}
      <div className="flex items-center gap-1 ml-auto">
        <button onClick={() => onPageChange(page - 1)} disabled={page <= 1}
          className="p-1.5 rounded text-[#9e9eb0] hover:text-[#e0e0e0] hover:bg-[#2d2d44]/40 disabled:opacity-30 disabled:cursor-not-allowed transition-colors">
          <ChevronLeft size={16} />
        </button>
        {pages.map((p, i) => (
          typeof p === 'number' ? (
            <button key={i} onClick={() => onPageChange(p)}
              className={`w-8 h-8 rounded text-xs font-medium transition-colors ${
                p === page ? 'bg-[#4fc3f7]/20 text-[#4fc3f7]' : 'text-[#9e9eb0] hover:text-[#e0e0e0] hover:bg-[#2d2d44]/30'
              }`}>{p}</button>
          ) : (
            <span key={i} className="px-1 text-[#9e9eb0] text-xs">...</span>
          )
        ))}
        <button onClick={() => onPageChange(page + 1)} disabled={page >= totalPages}
          className="p-1.5 rounded text-[#9e9eb0] hover:text-[#e0e0e0] hover:bg-[#2d2d44]/40 disabled:opacity-30 disabled:cursor-not-allowed transition-colors">
          <ChevronRight size={16} />
        </button>
      </div>
    </div>
  )
}
