import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Bot, Key, Eye, EyeOff, Loader2, Shield } from 'lucide-react';
import { useApiStore } from '../stores/apiStore';

export default function Login() {
  const navigate = useNavigate();
  const storeSetApiKey = useApiStore(s => s.setApiKey);
  const [apiKey, setApiKey] = useState('');
  const [showKey, setShowKey] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [mode, setMode] = useState<'key' | 'demo'>('key');

  const handleSubmit = async () => {
    if (mode === 'key' && !apiKey.trim()) { setError('API key required'); return; }
    setLoading(true); setError('');
    try {
      const res = await fetch('/api/auth/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ apiKey: apiKey.trim() || 'demo' }),
      });
      if (!res.ok) throw new Error('Invalid credentials');
      storeSetApiKey(apiKey.trim());
      navigate('/');
    } catch (e: any) { setError(e.message); }
    setLoading(false);
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-[#05050f] p-4">
      <div className="w-full max-w-sm space-y-6">
        <div className="text-center">
          <div className="w-16 h-16 rounded-2xl bg-[#4fc3f7]/10 flex items-center justify-center mx-auto mb-4 border border-[#4fc3f7]/20">
            <Bot size={32} className="text-[#4fc3f7]" />
          </div>
          <h1 className="text-2xl font-bold text-[#e0e0e0]">Aurelius</h1>
          <p className="text-sm text-[#9e9eb0] mt-1">Agent Operations Center</p>
        </div>

        <div className="aurelius-card p-6 space-y-4">
          <div className="flex gap-2 bg-[#0f0f1a] rounded-lg p-1">
            <button onClick={() => setMode('key')} className={`flex-1 py-2 text-xs font-medium rounded-md transition-colors ${mode === 'key' ? 'bg-[#4fc3f7]/10 text-[#4fc3f7]' : 'text-[#9e9eb0]'}`}>
              <Key size={12} className="inline mr-1" />API Key
            </button>
            <button onClick={() => setMode('demo')} className={`flex-1 py-2 text-xs font-medium rounded-md transition-colors ${mode === 'demo' ? 'bg-emerald-500/10 text-emerald-400' : 'text-[#9e9eb0]'}`}>
              Demo
            </button>
          </div>

          {mode === 'key' && (
            <div>
              <label className="text-xs text-[#9e9eb0] mb-1 block">API Key</label>
              <div className="relative">
                <input type={showKey ? 'text' : 'password'} value={apiKey} onChange={e => setApiKey(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handleSubmit()}
                  placeholder="sk-..." className="w-full bg-[#0f0f1a] border border-[#2d2d44] rounded-lg px-4 py-2.5 text-sm text-[#e0e0e0] placeholder:text-[#9e9eb0] focus:outline-none focus:border-[#4fc3f7]/50 pr-10" />
                <button onClick={() => setShowKey(!showKey)} className="absolute right-3 top-1/2 -translate-y-1/2 text-[#9e9eb0]">
                  {showKey ? <EyeOff size={14} /> : <Eye size={14} />}
                </button>
              </div>
            </div>
          )}

          {mode === 'demo' && (
            <div className="text-center py-3">
              <Shield size={24} className="mx-auto text-emerald-400 mb-2" />
              <p className="text-xs text-[#9e9eb0]">Explore the dashboard with demo data. No authentication required.</p>
            </div>
          )}

          {error && <p className="text-rose-400 text-xs text-center">{error}</p>}

          <button onClick={handleSubmit} disabled={loading}
            className="w-full bg-[#4fc3f7] hover:bg-[#4fc3f7]/90 text-[#05050f] font-semibold py-2.5 rounded-lg transition-colors text-sm flex items-center justify-center gap-2 disabled:opacity-50">
            {loading ? <Loader2 size={14} className="animate-spin" /> : null}
            {mode === 'demo' ? 'Enter Demo Mode' : 'Sign In'}
          </button>
        </div>

        <p className="text-center text-[10px] text-[#2d2d44]">Aurelius Systems · v1.0.0</p>
      </div>
    </div>
  );
}
