export const tokens = {
  colors: {
    primary: '#4fc3f7',
    success: '#34d399',
    warning: '#fbbf24',
    error: '#f87171',
    info: '#4fc3f7',
    bg: '#0f0f1a',
    surface: '#1a1a2e',
    border: '#2d2d44',
    text: '#e0e0e0',
    textSecondary: '#9e9eb0',
  },
  spacing: {
    xs: '0.25rem',
    sm: '0.5rem',
    md: '1rem',
    lg: '1.5rem',
    xl: '2rem',
  },
  radius: {
    sm: '0.375rem',
    md: '0.5rem',
    lg: '0.75rem',
    xl: '1rem',
  },
  font: {
    sans: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif",
    mono: "'SF Mono', 'Fira Code', monospace",
  },
  shadow: {
    card: '0 1px 3px rgba(0,0,0,0.3), 0 1px 2px rgba(0,0,0,0.2)',
  },
} as const

export type DesignToken = typeof tokens
