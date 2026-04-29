interface AreaChartProps<T extends Record<string, unknown> = Record<string, unknown>> {
  data: T[]
  xKey?: keyof T | string
  yKey?: keyof T | string
  color?: string
  fillOpacity?: number
  height?: number
}

export default function AreaChart<T extends Record<string, unknown>>({
  data,
  xKey,
  yKey,
  color = '#4fc3f7',
  fillOpacity = 0.2,
  height = 120,
}: AreaChartProps<T>) {
  if (data.length === 0) return null

  const normalizedData = data.map((item, index) => {
    const record = item as Record<string, unknown> & {
      label?: string
      name?: string
      value?: number
    }
    const labelValue = xKey ? record[xKey as string] : record.label ?? record.name ?? `${index + 1}`
    const valueValue = yKey ? record[yKey as string] : record.value ?? 0
    return { label: String(labelValue ?? `${index + 1}`), value: Number(valueValue ?? 0) }
  })

  const width = normalizedData.length * 20
  const maxValue = Math.max(...normalizedData.map((d) => d.value), 1)
  const padding = 4
  const chartHeight = height - padding * 2
  const chartWidth = width - padding * 2

  const points = normalizedData.map((d, i) => {
    const x = padding + (i / (normalizedData.length - 1 || 1)) * chartWidth
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
