/** @type {import('tailwindcss').Config} */
export default {
  content: [
    "./index.html",
    "./src/**/*.{js,ts,jsx,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        aurelius: {
          bg: '#0f0f1a',
          card: '#1a1a2e',
          border: '#2d2d44',
          text: '#e0e0e0',
          muted: '#a0a0b8',
          accent: '#4fc3f7',
          accentHover: '#3aa8d8',
        }
      }
    },
  },
  plugins: [],
}
