import { useState, useMemo, useCallback } from 'react'

interface UsePaginationOptions<T> {
  data: T[]
  pageSize?: number
  initialPage?: number
}

interface UsePaginationReturn<T> {
  page: number
  totalPages: number
  total: number
  pageData: T[]
  hasNext: boolean
  hasPrev: boolean
  goTo: (page: number) => void
  next: () => void
  prev: () => void
  first: () => void
  last: () => void
  setPageSize: (size: number) => void
}

export function usePagination<T>({ data, pageSize = 25, initialPage = 1 }: UsePaginationOptions<T>): UsePaginationReturn<T> {
  const [page, setPage] = useState(initialPage)
  const [size, setSize] = useState(pageSize)

  const totalPages = useMemo(() => Math.max(1, Math.ceil(data.length / size)), [data.length, size])

  const pageData = useMemo(() => {
    const start = (page - 1) * size
    return data.slice(start, start + size)
  }, [data, page, size])

  const safePage = useCallback((p: number) => Math.max(1, Math.min(p, totalPages)), [totalPages])

  return {
    page,
    totalPages,
    total: data.length,
    pageData,
    hasNext: page < totalPages,
    hasPrev: page > 1,
    goTo: (p) => setPage(safePage(p)),
    next: () => setPage((p) => safePage(p + 1)),
    prev: () => setPage((p) => safePage(p - 1)),
    first: () => setPage(1),
    last: () => setPage(totalPages),
    setPageSize: (newSize) => { setSize(newSize); setPage(1) },
  }
}
