import { Link } from 'react-router-dom'
import { ChevronRight, Home } from 'lucide-react'

interface Crumb {
  label: string
  href?: string
  icon?: React.ReactNode
}

interface BreadcrumbsProps {
  items: Crumb[]
}

export function Breadcrumbs({ items }: BreadcrumbsProps) {
  return (
    <nav aria-label="Breadcrumb" className="flex items-center gap-1.5 text-xs text-[#9e9eb0] mb-4">
      <Link to="/" className="hover:text-[#4fc3f7] transition-colors"><Home size={14} /></Link>
      {items.map((item, i) => (
        <span key={i} className="flex items-center gap-1.5">
          <ChevronRight size={12} className="text-[#2d2d44]" />
          {item.href ? (
            <Link to={item.href} className="hover:text-[#4fc3f7] transition-colors flex items-center gap-1">
              {item.icon}{item.label}
            </Link>
          ) : (
            <span className="text-[#e0e0e0] flex items-center gap-1">{item.icon}{item.label}</span>
          )}
        </span>
      ))}
    </nav>
  )
}
