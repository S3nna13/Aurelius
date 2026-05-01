import { useNavigate } from 'react-router-dom';
import { RefreshCw, Home, AlertTriangle } from 'lucide-react';

export default function ServerError() {
  const navigate = useNavigate();
  return (
    <div className="min-h-[60vh] flex items-center justify-center">
      <div className="text-center max-w-md">
        <div className="w-16 h-16 rounded-2xl bg-rose-500/10 flex items-center justify-center mx-auto mb-4 border border-rose-500/20">
          <AlertTriangle size={32} className="text-rose-400" />
        </div>
        <h1 className="text-xl font-bold text-[#e0e0e0] mb-2">Server Error</h1>
        <p className="text-sm text-[#9e9eb0] mb-2">Something went wrong on our end. Please try again.</p>
        <p className="text-xs text-[#2d2d44] mb-6 font-mono">HTTP 500 Internal Server Error</p>
        <div className="flex justify-center gap-3">
          <button onClick={() => window.location.reload()} className="aurelius-btn-outline flex items-center gap-2 text-sm"><RefreshCw size={14} /> Retry</button>
          <button onClick={() => navigate('/')} className="aurelius-btn-primary flex items-center gap-2 text-sm"><Home size={14} /> Dashboard</button>
        </div>
      </div>
    </div>
  );
}
