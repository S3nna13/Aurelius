import { useId } from 'react';

interface DataPoint {
  label: string;
  value: number;
}

interface LineChartProps {
  data: DataPoint[];
  width?: number;
  height?: number;
  color?: string;
  title?: string;
}

export default function LineChart({
  data,
  width = 600,
  height = 200,
  color = '#4fc3f7',
  title,
}: LineChartProps) {
  const uid = useId();

  if (data.length === 0) return null;

  const padding = { top: 20, right: 20, bottom: 30, left: 40 };
  const chartWidth = width - padding.left - padding.right;
  const chartHeight = height - padding.top - padding.bottom;

  const maxValue = Math.max(...data.map((d) => d.value), 1);
  const minValue = Math.min(...data.map((d) => d.value), 0);
  const range = maxValue - minValue || 1;

  const points = data.map((d, i) => {
    const x = padding.left + (i / (data.length - 1 || 1)) * chartWidth;
    const y = padding.top + chartHeight - ((d.value - minValue) / range) * chartHeight;
    return { x, y, label: d.label, value: d.value };
  });

  const pathD = points.map((p, i) => `${i === 0 ? 'M' : 'L'} ${p.x} ${p.y}`).join(' ');

  const areaD =
    pathD +
    ` L ${points[points.length - 1].x} ${padding.top + chartHeight}` +
    ` L ${points[0].x} ${padding.top + chartHeight} Z`;

  const yTicks = 5;
  const yTickValues = Array.from({ length: yTicks + 1 }, (_, i) =>
    Math.round(minValue + (range * i) / yTicks)
  );

  const gradientId = `lg-${uid}`;

  return (
    <div className="w-full">
      {title && <p className="text-xs font-bold text-[#9e9eb0] uppercase tracking-wider mb-2">{title}</p>}
      <svg viewBox={`0 0 ${width} ${height}`} className="w-full h-auto" preserveAspectRatio="xMidYMid meet">
        <defs>
          <linearGradient id={gradientId} x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor={color} stopOpacity="0.3" />
            <stop offset="100%" stopColor={color} stopOpacity="0" />
          </linearGradient>
        </defs>

        {yTickValues.map((tick, i) => {
          const y = padding.top + chartHeight - ((tick - minValue) / range) * chartHeight;
          return (
            <g key={i}>
              <line x1={padding.left} y1={y} x2={width - padding.right} y2={y} stroke="#2d2d44" strokeWidth="1" />
              <text x={padding.left - 8} y={y + 4} textAnchor="end" fill="#9e9eb0" fontSize="10">
                {tick}
              </text>
            </g>
          );
        })}

        <path d={areaD} fill={`url(#${gradientId})`} />
        <path d={pathD} fill="none" stroke={color} strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" />

        {points.map((p, i) => (
          <g key={i}>
            <circle cx={p.x} cy={p.y} r="4" fill={color} stroke="#1a1a2e" strokeWidth="2" />
            <title>{`${p.label}: ${p.value}`}</title>
          </g>
        ))}

        {points.filter((_, i) => i % Math.ceil(points.length / 6) === 0 || i === points.length - 1).map((p, i) => (
          <text key={i} x={p.x} y={height - 8} textAnchor="middle" fill="#9e9eb0" fontSize="10">
            {p.label}
          </text>
        ))}
      </svg>
    </div>
  );
}
