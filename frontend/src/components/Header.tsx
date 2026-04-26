import { useState, useEffect } from 'react';
import { Bell, Wifi, WifiOff, Command, Sun, Moon } from 'lucide-react';
import { Link } from 'react-router-dom';
import { useApi } from '../hooks/useApi';
import { useTheme } from '../context/ThemeContext';
import AutoRefreshControl from './AutoRefreshControl';

interface HeaderProps {
  onOpenPalette?: () => void;
}

export default function Header({ onOpenPalette }: HeaderProps) {
  const [online, setOnline] = useState(navigator.onLine);
  const { theme, toggleTheme } = useTheme();
  const { data: statsData } = useApi<{ unread: number }>('/notifications/stats', {
    refreshInterval: 10000,
  });
  const unreadCount = statsData?.unread ?? 0;

  useEffect(() => {
    const handleOnline = () => setOnline(true);
    const handleOffline = () => setOnline(false);
    window.addEventListener('online', handleOnline);
    window.addEventListener('offline', handleOffline);
    return () => {
      window.removeEventListener('online', handleOnline);
      window.removeEventListener('offline', handleOffline);
    };
  }, []);

  return (
    <header className="h-16 bg-aurelius-card border-b border-aurelius-border flex items-center justify-between px-6 md:pl-72">
      <div className="flex items-center gap-2">
        <h2 className="text-sm font-semibold text-aurelius-muted uppercase tracking-wider hidden md:block">
          Mission Control
        </h2>
      </div>

      <div className="flex items-center gap-4">
        <AutoRefreshControl />

        {/* Theme Toggle */}
        <button
          onClick={toggleTheme}
          className="p-2 rounded-lg text-aurelius-muted hover:text-aurelius-text hover:bg-aurelius-border/40 transition-colors"
          aria-label="Toggle theme"
        >
          {theme === 'dark' ? <Sun size={18} /> : <Moon size={18} />}
        </button>

        {/* Cmd+K hint */}
        <button
          onClick={onOpenPalette}
          className="hidden sm:flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs text-aurelius-muted bg-aurelius-bg border border-aurelius-border hover:border-aurelius-accent/30 transition-colors"
        >
          <Command size={12} />
          <span>Cmd+K</span>
        </button>

        {/* Connection Status */}
        <div
          className={`
            flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border
            ${online
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
              : 'bg-red-500/10 text-red-400 border-red-500/20'
            }
          `}
        >
          {online ? <Wifi size={14} /> : <WifiOff size={14} />}
          <span>{online ? 'Online' : 'Offline'}</span>
        </div>

        {/* Notification Bell */}
        <Link
          to="/notifications"
          className="relative p-2 rounded-lg text-aurelius-muted hover:text-aurelius-text hover:bg-aurelius-border/40 transition-colors"
          aria-label="Notifications"
        >
          <Bell size={20} />
          {unreadCount > 0 && (
            <span className="absolute top-1 right-1 flex h-4 min-w-[16px] items-center justify-center rounded-full bg-red-500 px-1 text-[10px] font-bold text-white">
              {unreadCount > 99 ? '99+' : unreadCount}
            </span>
          )}
        </Link>
      </div>
    </header>
  );
}
