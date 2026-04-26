import { createContext, useContext, useState, useCallback, useRef, type ReactNode } from 'react';
import { X, CheckCircle, AlertTriangle, Info } from 'lucide-react';
import { AnimatePresence, motion } from 'framer-motion';

type ToastType = 'success' | 'error' | 'warning' | 'info';

interface Toast {
  id: string;
  message: string;
  type: ToastType;
}

interface ToastContextValue {
  toast: (message: string, type?: ToastType) => void;
}

const ToastContext = createContext<ToastContextValue | null>(null);

export function useToast(): ToastContextValue {
  const ctx = useContext(ToastContext);
  if (!ctx) throw new Error('useToast must be used within ToastProvider');
  return ctx;
}

const config: Record<ToastType, { icon: typeof Info; classes: string }> = {
  success: { icon: CheckCircle, classes: 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20' },
  error: { icon: AlertTriangle, classes: 'bg-rose-500/10 text-rose-400 border-rose-500/20' },
  warning: { icon: AlertTriangle, classes: 'bg-amber-500/10 text-amber-400 border-amber-500/20' },
  info: { icon: Info, classes: 'bg-[#4fc3f7]/10 text-[#4fc3f7] border-[#4fc3f7]/20' },
};

let toastIdCounter = 0;
const MAX_TOASTS = 5;
const TOAST_DURATION = 4000;

export default function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);
  const timersRef = useRef<Record<string, number>>({});
  const pausedRef = useRef<Record<string, boolean>>({});

  const toast = useCallback((message: string, type: ToastType = 'info') => {
    const id = `toast-${++toastIdCounter}-${Date.now()}`;
    setToasts((prev) => {
      const next = [...prev, { id, message, type }];
      if (next.length > MAX_TOASTS) {
        const removed = next[0];
        window.clearTimeout(timersRef.current[removed.id]);
        delete timersRef.current[removed.id];
        delete pausedRef.current[removed.id];
        return next.slice(1);
      }
      return next;
    });
    timersRef.current[id] = window.setTimeout(() => {
      remove(id);
    }, TOAST_DURATION);
  }, []);

  const remove = (id: string) => {
    window.clearTimeout(timersRef.current[id]);
    delete timersRef.current[id];
    delete pausedRef.current[id];
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  const handleMouseEnter = (id: string) => {
    pausedRef.current[id] = true;
    window.clearTimeout(timersRef.current[id]);
  };

  const handleMouseLeave = (id: string) => {
    if (!pausedRef.current[id]) return;
    pausedRef.current[id] = false;
    timersRef.current[id] = window.setTimeout(() => {
      remove(id);
    }, TOAST_DURATION);
  };

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}
      <div className="fixed bottom-4 right-4 z-[100] space-y-2">
        <AnimatePresence>
          {toasts.map((t) => {
            const { icon: Icon, classes } = config[t.type];
            return (
              <motion.div
                key={t.id}
                layout
                initial={{ opacity: 0, y: 20, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                exit={{ opacity: 0, x: 20, scale: 0.95 }}
                transition={{ duration: 0.2 }}
                onMouseEnter={() => handleMouseEnter(t.id)}
                onMouseLeave={() => handleMouseLeave(t.id)}
                className={`flex items-center gap-3 px-4 py-3 rounded-lg border shadow-lg backdrop-blur-sm ${classes} min-w-[280px] max-w-md cursor-default`}
              >
                <Icon size={18} />
                <p className="text-sm font-medium flex-1">{t.message}</p>
                <button onClick={() => remove(t.id)} className="opacity-60 hover:opacity-100 transition-opacity">
                  <X size={14} />
                </button>
              </motion.div>
            );
          })}
        </AnimatePresence>
      </div>
    </ToastContext.Provider>
  );
}
