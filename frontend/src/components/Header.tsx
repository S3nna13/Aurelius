import { Bell, Wifi, WifiOff } from 'lucide-react';
import { Link } from 'react-router-dom';

interface HeaderProps {
  connected?: boolean;
  unreadCount?: number;
}

export default function Header({ connected = true, unreadCount = 3 }: HeaderProps) {
  return (
    <header className="h-16 bg-aurelius-card border-b border-aurelius-border flex items-center justify-between px-6 md:pl-72">
      <div className="flex items-center gap-2">
        <h2 className="text-sm font-semibold text-aurelius-muted uppercase tracking-wider hidden md:block">
          Mission Control
        </h2>
      </div>

      <div className="flex items-center gap-4">
        {/* Connection Status */}
        <div
          className={`
            flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium border
            ${connected
              ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
              : 'bg-red-500/10 text-red-400 border-red-500/20'
            }
          `}
        >
          {connected ? <Wifi size={14} /> : <WifiOff size={14} />}
          <span>{connected ? 'Online' : 'Offline'}</span>
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
