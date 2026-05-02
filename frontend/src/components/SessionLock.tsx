import { useState, useEffect, useCallback } from 'react';
import { Lock, Shield } from 'lucide-react';

interface SessionLockProps {
  timeoutMinutes?: number;
}

const INACTIVITY_EVENTS = ['mousedown', 'mousemove', 'keydown', 'scroll', 'touchstart'];

function isEnabled(): boolean {
  try {
    return localStorage.getItem('aurelius-auto-lock') === 'true';
  } catch {
    return false;
  }
}

export default function SessionLock({ timeoutMinutes = 5 }: SessionLockProps) {
  const [locked, setLocked] = useState(false);
  const [lastActivity, setLastActivity] = useState(() => Date.now());
  const [enabled, setEnabled] = useState(isEnabled);

  const resetActivity = useCallback(() => {
    setLastActivity(Date.now());
    if (locked) setLocked(false);
  }, [locked]);

  useEffect(() => {
    const check = () => setEnabled(isEnabled());
    const interval = setInterval(check, 5000);
    window.addEventListener('storage', check);
    return () => {
      clearInterval(interval);
      window.removeEventListener('storage', check);
    };
  }, []);

  useEffect(() => {
    if (!enabled) return;
    const events = INACTIVITY_EVENTS;
    events.forEach((e) => document.addEventListener(e, resetActivity));
    return () => events.forEach((e) => document.removeEventListener(e, resetActivity));
  }, [enabled, resetActivity]);

  useEffect(() => {
    if (!enabled) return;
    const interval = setInterval(() => {
      const inactiveMs = Date.now() - lastActivity;
      if (inactiveMs > timeoutMinutes * 60 * 1000 && !locked) {
        setLocked(true);
      }
    }, 5000);
    return () => clearInterval(interval);
  }, [enabled, lastActivity, locked, timeoutMinutes]);

  if (!enabled || !locked) return null;

  return (
    <div
      className="fixed inset-0 z-[70] flex flex-col items-center justify-center bg-aurelius-bg/95 backdrop-blur-md"
      onClick={resetActivity}
      onKeyDown={resetActivity}
      tabIndex={0}
    >
      <div className="flex flex-col items-center gap-4 animate-pulse">
        <div className="w-16 h-16 rounded-full bg-aurelius-accent/10 border border-aurelius-accent/30 flex items-center justify-center">
          <Lock size={32} className="text-aurelius-accent" />
        </div>
        <h2 className="text-xl font-bold text-aurelius-text">Session Locked</h2>
        <p className="text-sm text-aurelius-muted">Click or press any key to unlock</p>
      </div>
      <div className="absolute bottom-6 flex items-center gap-2 text-xs text-aurelius-muted">
        <Shield size={12} />
        Aurelius Security
      </div>
    </div>
  );
}
