import { useState, useCallback, useEffect, useRef } from 'react';
import { RefreshCw, Loader2, Clock } from 'lucide-react';

interface AutoRefreshControlProps {
  onRefresh?: () => void;
  interval?: number;
  loading?: boolean;
}

export default function AutoRefreshControl({ onRefresh, interval = 5000, loading }: AutoRefreshControlProps) {
  const [enabled, setEnabled] = useState(() => {
    try {
      return localStorage.getItem('aurelius-auto-refresh') !== 'false';
    } catch {
      return true;
    }
  });
  const [lastRefresh, setLastRefresh] = useState<Date>(new Date());
  const [now, setNow] = useState(() => Date.now());
  const mountedRef = useRef(false);

  const refresh = useCallback(() => {
    onRefresh?.();
    window.dispatchEvent(new CustomEvent('aurelius:auto-refresh', { detail: true }));
    setLastRefresh(new Date());
  }, [onRefresh]);

  useEffect(() => {
    try {
      localStorage.setItem('aurelius-auto-refresh', enabled ? 'true' : 'false');
    } catch {
      // Ignore storage failures.
    }
    if (mountedRef.current && enabled) {
      window.dispatchEvent(new CustomEvent('aurelius:auto-refresh', { detail: true }));
    }
    mountedRef.current = true;
  }, [enabled]);

  useEffect(() => {
    const id = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(id);
  }, []);

  const secondsSince = Math.max(0, Math.floor((now - lastRefresh.getTime()) / 1000));
  const intervalLabel = `${Math.max(1, Math.round(interval / 1000))}s`;

  return (
    <div className="flex items-center gap-2">
      {enabled && (
        <span className="text-[10px] text-[#9e9eb0] flex items-center gap-1">
          <Clock size={10} /> {secondsSince}s · Auto {intervalLabel}
        </span>
      )}
      <button onClick={refresh} disabled={loading}
        className="aurelius-btn-outline p-1.5 disabled:opacity-50" title="Refresh">
        {loading ? <Loader2 size={12} className="animate-spin" /> : <RefreshCw size={12} />}
      </button>
      <label className="flex items-center gap-1.5 cursor-pointer">
        <input type="checkbox" checked={enabled} onChange={() => setEnabled(!enabled)}
          className="w-3 h-3 rounded border-[#2d2d44] bg-[#0f0f1a] text-[#4fc3f7]" />
        <span className="text-[10px] text-[#9e9eb0]">Auto</span>
      </label>
    </div>
  );
}
