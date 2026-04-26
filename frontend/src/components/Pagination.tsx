import { ChevronLeft, ChevronRight } from 'lucide-react';

interface PaginationProps {
  currentPage: number;
  totalPages: number;
  onPageChange: (page: number) => void;
}

export default function Pagination({ currentPage, totalPages, onPageChange }: PaginationProps) {
  if (totalPages <= 1) return null;

  const getPages = () => {
    const pages: (number | string)[] = [];
    const showEllipsis = totalPages > 7;

    if (!showEllipsis) {
      for (let i = 1; i <= totalPages; i++) pages.push(i);
      return pages;
    }

    pages.push(1);
    if (currentPage > 3) pages.push('...');

    const start = Math.max(2, currentPage - 1);
    const end = Math.min(totalPages - 1, currentPage + 1);
    for (let i = start; i <= end; i++) pages.push(i);

    if (currentPage < totalPages - 2) pages.push('...');
    pages.push(totalPages);
    return pages;
  };

  return (
    <div className="flex items-center justify-center gap-1.5">
      <button
        onClick={() => onPageChange(currentPage - 1)}
        disabled={currentPage <= 1}
        className="p-1.5 rounded-lg text-aurelius-muted hover:text-aurelius-text hover:bg-aurelius-border/40 disabled:opacity-30 disabled:hover:bg-transparent transition-colors"
      >
        <ChevronLeft size={16} />
      </button>
      {getPages().map((p, i) =>
        typeof p === 'string' ? (
          <span key={`ellipsis-${i}`} className="px-2 text-xs text-aurelius-muted">
            {p}
          </span>
        ) : (
          <button
            key={p}
            onClick={() => onPageChange(p)}
            className={`min-w-[28px] h-7 px-1.5 rounded-lg text-xs font-medium transition-colors ${
              p === currentPage
                ? 'bg-aurelius-accent text-aurelius-bg'
                : 'text-aurelius-muted hover:text-aurelius-text hover:bg-aurelius-border/40'
            }`}
          >
            {p}
          </button>
        )
      )}
      <button
        onClick={() => onPageChange(currentPage + 1)}
        disabled={currentPage >= totalPages}
        className="p-1.5 rounded-lg text-aurelius-muted hover:text-aurelius-text hover:bg-aurelius-border/40 disabled:opacity-30 disabled:hover:bg-transparent transition-colors"
      >
        <ChevronRight size={16} />
      </button>
    </div>
  );
}
