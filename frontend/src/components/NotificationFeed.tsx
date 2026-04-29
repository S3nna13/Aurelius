import { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Bell, X, CheckCircle, AlertTriangle, Info, AlertCircle, Bot } from 'lucide-react';

interface Notification {
  id: string; title: string; body: string; type: 'success' | 'error' | 'warning' | 'info' | 'agent';
  timestamp: number;
}

const ICONS = {
  success: CheckCircle, error: AlertCircle, warning: AlertTriangle, info: Info, agent: Bot,
};
const COLORS = {
  success: 'border-emerald-500/30 bg-emerald-500/5 text-emerald-400',
  error: 'border-rose-500/30 bg-rose-500/5 text-rose-400',
  warning: 'border-amber-500/30 bg-amber-500/5 text-amber-400',
  info: 'border-[#4fc3f7]/30 bg-[#4fc3f7]/5 text-[#4fc3f7]',
  agent: 'border-violet-500/30 bg-violet-500/5 text-violet-400',
};

export default function NotificationFeed() {
  const [notifications, setNotifications] = useState<Notification[]>([]);

  const remove = useCallback((id: string) => {
    setNotifications(prev => prev.filter(n => n.id !== id));
  }, []);

  useEffect(() => {
    const es = new EventSource('/api/notifications/stream');
    es.onmessage = (e) => {
      try {
        const data = JSON.parse(e.data);
        const n: Notification = {
          id: Date.now().toString(36) + Math.random().toString(36).slice(2, 4),
          title: data.title || 'Notification',
          body: data.body || '',
          type: data.priority === 'critical' ? 'error' : data.priority === 'high' ? 'warning' : data.type || 'info',
          timestamp: Date.now(),
        };
        setNotifications(prev => [n, ...prev].slice(0, 5));
        setTimeout(() => remove(n.id), 5000);
      } catch {}
    };
    return () => es.close();
  }, [remove]);

  return (
    <div className="fixed top-4 right-4 z-50 w-80 space-y-2 pointer-events-none">
      <AnimatePresence>
        {notifications.map(n => {
          const Icon = ICONS[n.type] || Info;
          const color = COLORS[n.type] || COLORS.info;
          return (
            <motion.div key={n.id} initial={{ opacity: 0, x: 80, scale: 0.95 }} animate={{ opacity: 1, x: 0, scale: 1 }}
              exit={{ opacity: 0, x: 80, scale: 0.95 }} transition={{ type: 'spring', stiffness: 300, damping: 25 }}
              className={`pointer-events-auto rounded-xl border p-3 backdrop-blur-md ${color} bg-opacity-80`}
            >
              <div className="flex items-start gap-3">
                <Icon size={16} className="mt-0.5 shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-xs font-semibold truncate">{n.title}</p>
                  {n.body && <p className="text-[10px] opacity-80 mt-0.5 line-clamp-2">{n.body}</p>}
                </div>
                <button onClick={() => remove(n.id)} className="opacity-60 hover:opacity-100"><X size={12} /></button>
              </div>
            </motion.div>
          );
        })}
      </AnimatePresence>
    </div>
  );
}
