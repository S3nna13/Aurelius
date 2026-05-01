const CHART_COLORS = [
  '#4fc3f7', '#34d399', '#fbbf24', '#f87171', '#a78bfa',
  '#fb923c', '#2dd4bf', '#f472b6', '#818cf8', '#facc15',
  '#38bdf8', '#4ade80', '#c084fc', '#fb7185', '#eab308',
]

const STATUS_COLORS: Record<string, string> = {
  active: '#34d399',
  running: '#4fc3f7',
  idle: '#fbbf24',
  paused: '#fb923c',
  error: '#f87171',
  failed: '#f87171',
  success: '#34d399',
  warning: '#fbbf24',
  info: '#4fc3f7',
  healthy: '#34d399',
  degraded: '#fbbf24',
  unhealthy: '#f87171',
}

export function getChartColor(index: number): string {
  return CHART_COLORS[index % CHART_COLORS.length]
}

export function getStatusColor(status: string): string {
  return STATUS_COLORS[status.toLowerCase()] || '#9e9eb0'
}

export function getGradient(canvas: HTMLCanvasElement, color: string): CanvasGradient {
  const ctx = canvas.getContext('2d')!
  const gradient = ctx.createLinearGradient(0, 0, 0, canvas.height)
  gradient.addColorStop(0, `${color}40`)
  gradient.addColorStop(1, `${color}05`)
  return gradient
}

export function hexToRgba(hex: string, alpha: number): string {
  const r = parseInt(hex.slice(1, 3), 16)
  const g = parseInt(hex.slice(3, 5), 16)
  const b = parseInt(hex.slice(5, 7), 16)
  return `rgba(${r}, ${g}, ${b}, ${alpha})`
}

export function hexToRgb(hex: string): { r: number; g: number; b: number } {
  return {
    r: parseInt(hex.slice(1, 3), 16),
    g: parseInt(hex.slice(3, 5), 16),
    b: parseInt(hex.slice(5, 7), 16),
  }
}

export function parseColor(input: string): string {
  if (input.startsWith('#')) return input
  if (input.startsWith('rgb')) return input
  const named: Record<string, string> = {
    red: '#f87171', green: '#34d399', blue: '#4fc3f7',
    yellow: '#fbbf24', purple: '#a78bfa', orange: '#fb923c',
    pink: '#f472b6', cyan: '#2dd4bf', teal: '#2dd4bf',
  }
  return named[input.toLowerCase()] || '#4fc3f7'
}
