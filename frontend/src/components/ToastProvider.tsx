import { createContext, useContext, useState, useCallback, type ReactNode } from 'react';
import { X, CheckCircle, AlertTriangle, Info } from 'lucide-react';

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

export default function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const toast = useCallback((message: string, type: ToastType = 'info') => {
    const id = Math.random().toString(36).slice(2);
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  }, []);

  const remove = (id: string) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  return (
    <ToastContext.Provider value={{ toast }}>
      {children}
      <div className="fixed bottom-4 right-4 z-[100] space-y-2">
        {toasts.map((t) => {
          const { icon: Icon, classes } = config[t.type];
          return (
            <div
              key={t.id}
              className={`flex items-center gap-3 px-4 py-3 rounded-lg border shadow-lg backdrop-blur-sm ${classes} min-w-[280px] max-w-md`}
            >
              <Icon size={18} />
              <p className="text-sm font-medium flex-1">{t.message}</p>
              <button onClick={() => remove(t.id)} className="opacity-60 hover:opacity-100">
                <X size={14} />
              </button>
            </div>
          );
        })}
      </div>
    </ToastContext.Provider>
  );
}
