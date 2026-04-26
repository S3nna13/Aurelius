/** @type {import('tailwindcss').Config} */
export default {
  darkMode: 'class',
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        aurelius: {
          bg: 'var(--aurelius-bg)',
          card: 'var(--aurelius-card)',
          border: 'var(--aurelius-border)',
          text: 'var(--aurelius-text)',
          muted: 'var(--aurelius-muted)',
          accent: 'var(--aurelius-accent)',
          accentHover: 'var(--aurelius-accent-hover)',
          surface: 'var(--aurelius-surface)',
          surfaceHover: 'var(--aurelius-surface-hover)',
        }
      }
    },
  },
  plugins: [],
}
