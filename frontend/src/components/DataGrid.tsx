import { useState, useMemo, useCallback, type ReactNode } from 'react'
import { ChevronUp, ChevronDown, ChevronsUpDown, Search } from 'lucide-react'
import { Pagination } from './ui/Pagination'
import { Skeleton } from './Skeleton'

export interface DataGridColumn<T> {
  key: string
  header: string
  render?: (item: T) => ReactNode
  sortable?: boolean
  filterable?: boolean
  width?: string
  align?: 'left' | 'center' | 'right'
  cellClass?: string
}

interface DataGridProps<T> {
  columns: DataGridColumn<T>[]
  data: T[]
  keyField: keyof T | ((item: T) => string)
  loading?: boolean
  emptyMessage?: string
  pageSize?: number
  searchable?: boolean
  searchPlaceholder?: string
  onRowClick?: (item: T) => void
  compact?: boolean
}

export function DataGrid<T extends Record<string, unknown>>({
  columns, data, keyField, loading, emptyMessage = 'No data',
  pageSize = 25, searchable, searchPlaceholder = 'Filter...',
  onRowClick, compact = false,
}: DataGridProps<T>) {
  const [sortColumn, setSortColumn] = useState<string>('')
  const [sortDirection, setSortDirection] = useState<'asc' | 'desc'>('asc')
  const [searchQuery, setSearchQuery] = useState('')
  const [page, setPage] = useState(1)

  const getKey = useCallback((item: T): string => {
    if (typeof keyField === 'function') return keyField(item)
    return String(item[keyField])
  }, [keyField])

  const filtered = useMemo(() => {
    if (!searchQuery) return data
    const q = searchQuery.toLowerCase()
    return data.filter((item) =>
      columns.some((col) => {
        const val = item[col.key]
        return val !== null && val !== undefined && String(val).toLowerCase().includes(q)
      })
    )
  }, [data, searchQuery, columns])

  const sorted = useMemo(() => {
    if (!sortColumn) return filtered
    return [...filtered].sort((a, b) => {
      const aVal = a[sortColumn]
      const bVal = b[sortColumn]
      if (aVal == null) return 1
      if (bVal == null) return -1
      const cmp = typeof aVal === 'number' ? aVal - (bVal as number) : String(aVal).localeCompare(String(bVal))
      return sortDirection === 'asc' ? cmp : -cmp
    })
  }, [filtered, sortColumn, sortDirection])

  const totalPages = Math.max(1, Math.ceil(sorted.length / pageSize))
  const pageData = useMemo(() => {
    const start = (page - 1) * pageSize
    return sorted.slice(start, start + pageSize)
  }, [sorted, page, pageSize])

  const handleSort = (col: DataGridColumn<T>) => {
    if (!col.sortable) return
    if (sortColumn === col.key) setSortDirection((d) => (d === 'asc' ? 'desc' : 'asc'))
    else { setSortColumn(col.key); setSortDirection('asc') }
    setPage(1)
  }

  const SortIcon = ({ column }: { column: string }) => {
    if (sortColumn !== column) return <ChevronsUpDown size={12} className="text-[#9e9eb0]/40" />
    return sortDirection === 'asc' ? <ChevronUp size={12} className="text-[#4fc3f7]" /> : <ChevronDown size={12} className="text-[#4fc3f7]" />
  }

  return (
    <div className="space-y-3">
      {searchable && (
        <div className="relative max-w-xs">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#9e9eb0]" />
          <input
            type="text"
            value={searchQuery}
            onChange={(e) => { setSearchQuery(e.target.value); setPage(1) }}
            placeholder={searchPlaceholder}
            className="w-full bg-[#0f0f1a] border border-[#2d2d44] rounded-lg pl-9 pr-3 py-1.5 text-sm text-[#e0e0e0] placeholder:text-[#9e9eb0] focus:outline-none focus:border-[#4fc3f7]"
          />
        </div>
      )}

      <div className="overflow-x-auto rounded-lg border border-[#2d2d44]/50">
        <table className="w-full text-sm">
          <thead>
            <tr className="bg-[#0f0f1a]/80 text-[#9e9eb0] text-[10px] uppercase tracking-wider">
              {columns.map((col) => (
                <th
                  key={col.key}
                  className={`py-3 px-3 ${col.align === 'center' ? 'text-center' : col.align === 'right' ? 'text-right' : 'text-left'} ${col.sortable ? 'cursor-pointer hover:text-[#e0e0e0] select-none' : ''}`}
                  style={col.width ? { width: col.width } : undefined}
                  onClick={() => handleSort(col)}
                >
                  <div className={`flex items-center gap-1 ${col.align === 'center' ? 'justify-center' : col.align === 'right' ? 'justify-end' : ''}`}>
                    {col.header}
                    {col.sortable && <SortIcon column={col.key} />}
                  </div>
                </th>
              ))}
            </tr>
          </thead>
          <tbody>
            {loading ? (
              Array.from({ length: 5 }).map((_, i) => (
                <tr key={i} className="border-t border-[#2d2d44]/30">
                  {columns.map((col) => (
                    <td key={col.key} className="py-3 px-3"><Skeleton height="h-4" width="w-3/4" /></td>
                  ))}
                </tr>
              ))
            ) : pageData.length === 0 ? (
              <tr>
                <td colSpan={columns.length} className="text-center py-8 text-[#9e9eb0] text-sm">{searchQuery ? `No results for "${searchQuery}"` : emptyMessage}</td>
              </tr>
            ) : (
              pageData.map((item) => (
                <tr
                  key={getKey(item)}
                  className={`border-t border-[#2d2d44]/30 hover:bg-[#0f0f1a]/50 transition-colors ${onRowClick ? 'cursor-pointer' : ''}`}
                  onClick={() => onRowClick?.(item)}
                >
                  {columns.map((col) => (
                    <td
                      key={col.key}
                      className={`${compact ? 'py-2' : 'py-3'} px-3 ${col.align === 'center' ? 'text-center' : col.align === 'right' ? 'text-right' : 'text-left'} text-[#e0e0e0] ${col.cellClass || ''}`}
                    >
                      {col.render ? col.render(item) : String(item[col.key] ?? '')}
                    </td>
                  ))}
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {!loading && totalPages > 1 && (
        <Pagination page={page} totalPages={totalPages} onPageChange={setPage} total={sorted.length} />
      )}
    </div>
  )
}
