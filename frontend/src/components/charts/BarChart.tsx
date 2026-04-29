interface BarChartProps<T extends Record<string, unknown> = Record<string, unknown>> {
  data: T[];
  xKey?: keyof T | string;
  yKey?: keyof T | string;
  color?: string;
  width?: number;
  height?: number;
  title?: string;
}

export default function BarChart<T extends Record<string, unknown>>({
  data,
  xKey,
  yKey,
  color = '#4fc3f7',
  width = 400,
  height = 200,
  title,
}: BarChartProps<T>) {
  if (data.length === 0) return null;

  const normalizedData = data.map((item, index) => {
    const record = item as Record<string, unknown> & {
      label?: string;
      name?: string;
      value?: number;
    };
    const labelValue = xKey ? record[xKey as string] : record.label ?? record.name ?? `${index + 1}`;
    const valueValue = yKey ? record[yKey as string] : record.value ?? 0;
    return {
      label: String(labelValue ?? `${index + 1}`),
      value: Number(valueValue ?? 0),
      color,
    };
  });

  const padding = { top: 20, right: 20, bottom: 40, left: 40 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const maxValue = Math.max(...normalizedData.map((d) => d.value), 1);
  const barWidth = (chartWidth / normalizedData.length) * 0.6;
  const gap = (chartWidth / normalizedData.length) * 0.4;

  const yTicks = 5;
  const yTickValues = Array.from({ length: yTicks + 1 }, (_, i) =>
    Math.round((maxValue * i) / yTicks)
  );

  return (
    <div className="w-full">
      {title && <p className="text-xs font-bold text-[#9e9eb0] uppercase tracking-wider mb-2">{title}</p>}
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto" preserveAspectRatio="xMidYMid meet">
        {yTickValues.map((tick, i) => {
          const y = padding.top + chartHeight - (tick / maxValue) * chartHeight;
          return (
            <g key={i}>
              <line x1={padding.left} y1={y} x2={width - padding.right} y2={y} stroke="#2d2d44" strokeWidth="1" />
              <text x={padding.left - 8} y={y + 4} textAnchor="end" fill="#9e9eb0" fontSize="10">
                {tick}
              </text>
            </g>
          );
        })}

        {normalizedData.map((d, i) => {
          const barHeight = (d.value / maxValue) * chartHeight;
          const x = padding.left + i * (barWidth + gap) + gap / 2;
          const y = padding.top + chartHeight - barHeight;
          const color = d.color || '#4fc3f7';
          return (
            <g key={i}>
              <rect
                x={x}
                y={y}
                width={barWidth}
                height={barHeight}
                fill={color}
                rx="3"
                opacity="0.8"
              />
              <text x={x + barWidth / 2} y={y - 6} textAnchor="middle" fill="#e0e0e0" fontSize="10" fontWeight="bold">
                {d.value}
              </text>
              <text x={x + barWidth / 2} y={height - 8} textAnchor="middle" fill="#9e9eb0" fontSize="10">
                {d.label}
              </text>
              <title>{`${d.label}: ${d.value}`}</title>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
