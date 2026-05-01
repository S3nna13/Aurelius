interface DonutSegment {
  label?: string;
  name?: string;
  value: number;
  color?: string;
}

interface DonutChartProps {
  data: DonutSegment[];
  size?: number;
  title?: string;
}

export default function DonutChart({ data, size = 160, title }: DonutChartProps) {
  if (data.length === 0) return null;

  const palette = ['#4fc3f7', '#a78bfa', '#34d399', '#f59e0b', '#f43f5e', '#22d3ee', '#fb7185', '#84cc16'];
  const segments = data.map((segment, index) => ({
    label: segment.label ?? segment.name ?? `Segment ${index + 1}`,
    value: segment.value,
    color: segment.color ?? palette[index % palette.length],
  }));

  const total = segments.reduce((sum, d) => sum + d.value, 0) || 1;
  const strokeWidth = 24;
  const radius = (size - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const center = size / 2;

  return (
    <div className="flex flex-col items-center">
      {title && <p className="text-xs font-bold text-[#9e9eb0] uppercase tracking-wider mb-2">{title}</p>}
      <div className="flex items-center gap-4">
        <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
          {segments.map((d, i) => {
            const segmentLength = (d.value / total) * circumference;
            const dashArray = `${segmentLength} ${circumference - segmentLength}`;
            const dashOffset = -segments
              .slice(0, i)
              .reduce((sum, segment) => sum + (segment.value / total) * circumference, 0);
            return (
              <circle
                key={i}
                cx={center}
                cy={center}
                r={radius}
                fill="none"
                stroke={d.color}
                strokeWidth={strokeWidth}
                strokeDasharray={dashArray}
                strokeDashoffset={dashOffset}
                transform={`rotate(-90 ${center} ${center})`}
                strokeLinecap="round"
              >
                <title>{`${d.label}: ${d.value} (${Math.round((d.value / total) * 100)}%)`}</title>
              </circle>
            );
          })}
          <text x={center} y={center - 4} textAnchor="middle" fill="#e0e0e0" fontSize="18" fontWeight="bold">
            {total}
          </text>
          <text x={center} y={center + 12} textAnchor="middle" fill="#9e9eb0" fontSize="10">
            Total
          </text>
        </svg>
        <div className="space-y-1.5">
          {segments.map((d, i) => (
            <div key={i} className="flex items-center gap-2 text-xs">
              <span className="w-2.5 h-2.5 rounded-full shrink-0" style={{ backgroundColor: d.color }} />
              <span className="text-[#e0e0e0]">{d.label}</span>
              <span className="text-[#9e9eb0] ml-auto">{Math.round((d.value / total) * 100)}%</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
