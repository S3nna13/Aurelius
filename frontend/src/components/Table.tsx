import { type ReactNode } from 'react'
import { ChevronUp, ChevronDown, ChevronsUpDown } from 'lucide-react'

export interface Column<T> {
  key: string
  header: string
  render?: (item: T) => ReactNode
  sortable?: boolean
  width?: string
  align?: 'left' | 'center' | 'right'
}

interface TableProps<T> {
  columns: Column<T>[]
  data: T[]
  keyField: keyof T | ((item: T) => string)
  loading?: boolean
  emptyMessage?: string
  sortColumn?: string
  sortDirection?: 'asc' | 'desc'
  onSort?: (column: string) => void
  onRowClick?: (item: T) => void
  compact?: boolean
}

export function Table<T extends Record<string, unknown>>({
  columns, data, keyField, loading, emptyMessage = 'No data',
  sortColumn, sortDirection, onSort, onRowClick, compact = false,
}: TableProps<T>) {
  const getKey = (item: T): string => {
    if (typeof keyField === 'function') return keyField(item)
    return String(item[keyField])
  }

  const SortIcon = ({ column }: { column: string }) => {
    if (sortColumn !== column) return <ChevronsUpDown size={12} className="text-[#9e9eb0]/40" />
    return sortDirection === 'asc' ? <ChevronUp size={12} /> : <ChevronDown size={12} />
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-[#9e9eb0] text-[10px] uppercase tracking-wider border-b border-[#2d2d44]">
            {columns.map((col) => (
              <th
                key={col.key}
                className={`py-3 px-3 ${col.align === 'center' ? 'text-center' : col.align === 'right' ? 'text-right' : 'text-left'} ${col.sortable ? 'cursor-pointer hover:text-[#e0e0e0] select-none' : ''} ${col.width ? `w-[${col.width}]` : ''}`}
                onClick={() => col.sortable && onSort?.(col.key)}
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
          {data.map((item) => (
            <tr
              key={getKey(item)}
              className={`border-b border-[#2d2d44]/30 hover:bg-[#0f0f1a]/40 transition-colors ${onRowClick ? 'cursor-pointer' : ''}`}
              onClick={() => onRowClick?.(item)}
            >
              {columns.map((col) => (
                <td key={col.key} className={`py-3 px-3 ${compact ? 'py-2' : 'py-3'} ${col.align === 'center' ? 'text-center' : col.align === 'right' ? 'text-right' : 'text-left'} text-[#e0e0e0]`}>
                  {col.render ? col.render(item) : String(item[col.key] ?? '')}
                </td>
              ))}
            </tr>
          ))}
          {data.length === 0 && !loading && (
            <tr><td colSpan={columns.length} className="text-center py-8 text-[#9e9eb0]">{emptyMessage}</td></tr>
          )}
        </tbody>
      </table>
    </div>
  )
}
