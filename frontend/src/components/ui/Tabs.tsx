import { useState, type ReactNode } from 'react'

interface Tab {
  id: string
  label: string
  icon?: ReactNode
  badge?: number
}

interface TabsProps {
  tabs: Tab[]
  defaultTab?: string
  onChange?: (tabId: string) => void
  children: (activeTab: string) => ReactNode
}

export function Tabs({ tabs, defaultTab, onChange, children }: TabsProps) {
  const [activeTab, setActiveTab] = useState(defaultTab || tabs[0]?.id || '')

  const handleChange = (id: string) => {
    setActiveTab(id)
    onChange?.(id)
  }

  return (
    <div>
      <div className="flex gap-1 bg-[#0f0f1a] rounded-lg border border-[#2d2d44] p-1 overflow-x-auto">
        {tabs.map((tab) => (
          <button key={tab.id} onClick={() => handleChange(tab.id)}
            className={`flex items-center gap-1.5 px-3 py-1.5 text-xs rounded font-medium transition-colors whitespace-nowrap ${
              activeTab === tab.id
                ? 'bg-[#4fc3f7]/20 text-[#4fc3f7] shadow-sm'
                : 'text-[#9e9eb0] hover:text-[#e0e0e0] hover:bg-[#2d2d44]/30'
            }`}>
            {tab.icon && <span className="shrink-0">{tab.icon}</span>}
            {tab.label}
            {tab.badge !== undefined && (
              <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded-full ${
                activeTab === tab.id ? 'bg-[#4fc3f7]/30 text-[#4fc3f7]' : 'bg-[#2d2d44]/40 text-[#9e9eb0]'
              }`}>{tab.badge}</span>
            )}
          </button>
        ))}
      </div>
      <div className="mt-4">{children(activeTab)}</div>
    </div>
  )
}
