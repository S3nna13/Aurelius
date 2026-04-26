import { useState, useEffect } from 'react';
import { RefreshCw, Pause } from 'lucide-react';

const STORAGE_KEY = 'aurelius-auto-refresh';

export function useAutoRefresh() {
  const [enabled, setEnabled] = useState(() => {
    try {
      const saved = localStorage.getItem(STORAGE_KEY);
      return saved !== 'false';
    } catch {
      return true;
    }
  });

  useEffect(() => {
    localStorage.setItem(STORAGE_KEY, String(enabled));
    window.dispatchEvent(new CustomEvent('aurelius:auto-refresh', { detail: enabled }));
  }, [enabled]);

  return { enabled, setEnabled };
}

export default function AutoRefreshControl() {
  const { enabled, setEnabled } = useAutoRefresh();

  return (
    <button
      onClick={() => setEnabled(!enabled)}
      className={`flex items-center gap-1.5 px-2 py-1 rounded-lg text-xs font-medium border transition-colors ${
        enabled
          ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
          : 'bg-aurelius-bg text-aurelius-muted border-aurelius-border hover:border-aurelius-accent/30'
      }`}
      title={enabled ? 'Auto-refresh on' : 'Auto-refresh paused'}
    >
      {enabled ? <RefreshCw size={12} /> : <Pause size={12} />}
      {enabled ? 'Live' : 'Paused'}
    </button>
  );
}
