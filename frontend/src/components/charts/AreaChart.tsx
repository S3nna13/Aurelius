interface AreaPoint {
  label: string
  value: number
}

interface AreaChartProps {
  data: AreaPoint[]
  color?: string
  fillOpacity?: number
  height?: number
}

export default function AreaChart({ data, color = '#4fc3f7', fillOpacity = 0.2, height = 120 }: AreaChartProps) {
  if (data.length === 0) return null

  const width = data.length * 20
  const maxValue = Math.max(...data.map((d) => d.value), 1)
  const padding = 4
  const chartHeight = height - padding * 2
  const chartWidth = width - padding * 2

  const points = data.map((d, i) => {
    const x = padding + (i / (data.length - 1 || 1)) * chartWidth
    const y = padding + chartHeight - (d.value / maxValue) * chartHeight
    return { x, y, ...d }
  })

  const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'}${p.x},${p.y}`).join(' ')
  const areaD = `${pathD} L${points[points.length - 1].x},${padding + chartHeight} L${points[0].x},${padding + chartHeight} Z`

  return (
    <svg width="100%" height={height} viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none" className="overflow-visible">
      <defs>
        <linearGradient id={`area-fill-${color.replace('#', '')}`} x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity={fillOpacity} />
          <stop offset="100%" stopColor={color} stopOpacity={0} />
        </linearGradient>
      </defs>
      <path d={areaD} fill={`url(#area-fill-${color.replace('#', '')})`} />
      <path d={pathD} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />
      {points.filter((_, i) => i === points.length - 1).map((p) => (
        <circle key="last" cx={p.x} cy={p.y} r="3" fill={color} />
      ))}
    </svg>
  )
}
