import { useState, useEffect } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Wifi, WifiOff, Activity } from 'lucide-react';

export default function ConnectionStatus() {
  const [online, setOnline] = useState(navigator.onLine);
  const [latency, setLatency] = useState<number | null>(null);
  const [show, setShow] = useState(false);

  useEffect(() => {
    const goOnline = () => { setOnline(true); setShow(true); setTimeout(() => setShow(false), 3000); };
    const goOffline = () => { setOnline(false); setShow(true); };
    window.addEventListener('online', goOnline);
    window.addEventListener('offline', goOffline);

    const ping = setInterval(async () => {
      const start = Date.now();
      try {
        await fetch('/api/health', { method: 'HEAD' });
        setLatency(Date.now() - start);
      } catch { setLatency(null); }
    }, 30000);

    return () => { window.removeEventListener('online', goOnline); window.removeEventListener('offline', goOffline); clearInterval(ping); };
  }, []);

  return (
    <AnimatePresence>
      {(show || !online) && (
        <motion.div initial={{ opacity: 0, y: -20 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -20 }}
          className={`fixed top-0 left-0 right-0 z-[60] py-1.5 text-center text-[10px] font-medium ${
            online ? 'bg-emerald-500/10 text-emerald-400 border-b border-emerald-500/20' : 'bg-rose-500/10 text-rose-400 border-b border-rose-500/20'
          }`}
        >
          {online ? (
            <span className="flex items-center justify-center gap-1.5">
              <Wifi size={12} /> Connected {latency !== null && `(${latency}ms)`}
            </span>
          ) : (
            <span className="flex items-center justify-center gap-1.5">
              <WifiOff size={12} /> Offline — reconnecting...
            </span>
          )}
        </motion.div>
      )}
    </AnimatePresence>
  );
}
