import { useState, useEffect } from 'react';
import { Monitor, Moon, Sun } from 'lucide-react';

type Theme = 'dark' | 'light' | 'system';

export default function ThemeSwitcher() {
  const [theme, setTheme] = useState<Theme>(() => (localStorage.getItem('theme') as Theme) || 'dark');

  useEffect(() => {
    const root = document.documentElement;
    if (theme === 'dark' || (theme === 'system' && window.matchMedia('(prefers-color-scheme: dark)').matches)) {
      root.classList.add('dark');
    } else {
      root.classList.remove('dark');
    }
    localStorage.setItem('theme', theme);
  }, [theme]);

  const themes: { key: Theme; icon: typeof Sun; label: string }[] = [
    { key: 'dark', icon: Moon, label: 'Dark' },
    { key: 'light', icon: Sun, label: 'Light' },
    { key: 'system', icon: Monitor, label: 'System' },
  ];

  return (
    <div className="flex gap-1 bg-[#0f0f1a] rounded-lg p-0.5 border border-[#2d2d44]">
      {themes.map(t => {
        const Icon = t.icon;
        const active = theme === t.key;
        return (
          <button key={t.key} onClick={() => setTheme(t.key)}
            className={`p-1.5 rounded-md transition-colors ${active ? 'bg-[#4fc3f7]/10 text-[#4fc3f7]' : 'text-[#9e9eb0] hover:text-[#e0e0e0]'}`}
            title={t.label}
          >
            <Icon size={14} />
          </button>
        );
      })}
    </div>
  );
}
