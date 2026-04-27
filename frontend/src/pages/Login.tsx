import { useState } from 'react'
import { useNavigate } from 'react-router-dom'
import { KeyRound, Loader2, Shield, Eye, EyeOff, Bot } from 'lucide-react'
import { useAuth } from '../hooks/useAuth'
import { useToast } from '../components/ToastProvider'

export default function Login() {
  const [apiKey, setApiKey] = useState('')
  const [showKey, setShowKey] = useState(false)
  const [mode, setMode] = useState<'login' | 'register'>('login')
  const [username, setUsername] = useState('')
  const { authenticated, loading, error, login } = useAuth()
  const navigate = useNavigate()
  const { toast } = useToast()

  if (authenticated) {
    navigate('/')
    return null
  }

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!apiKey.trim()) return
    const ok = await login(apiKey.trim())
    if (ok) {
      toast('Authenticated successfully', 'success')
      navigate('/')
    } else {
      toast('Invalid API key', 'error')
    }
  }

  const handleRegister = async (e: React.FormEvent) => {
    e.preventDefault()
    if (!username.trim()) return
    try {
      const res = await fetch('/api/auth/register', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ username: username.trim() }),
      })
      const data = await res.json()
      if (data.success) {
        const ok = await login(data.apiKey)
        if (ok) {
          toast(`Registered as ${data.user.username}`, 'success')
          navigate('/')
        }
      } else {
        toast(data.error || 'Registration failed', 'error')
      }
    } catch {
      toast('Registration failed', 'error')
    }
  }

  const fillDevKey = () => setApiKey('dev-key')

  return (
    <div className="min-h-screen bg-[#0f0f1a] flex items-center justify-center p-4">
      <div className="w-full max-w-md space-y-8">
        <div className="text-center">
          <div className="w-16 h-16 mx-auto mb-4 rounded-2xl bg-[#4fc3f7]/15 border border-[#4fc3f7]/30 flex items-center justify-center">
            <Bot size={32} className="text-[#4fc3f7]" />
          </div>
          <h1 className="text-2xl font-bold text-[#e0e0e0]">Aurelius</h1>
          <p className="text-sm text-[#9e9eb0] mt-1">Mission Control</p>
        </div>

        <div className="bg-[#1a1a2e] border border-[#2d2d44] rounded-xl p-8 space-y-6">
          <div className="flex gap-1 bg-[#0f0f1a] rounded-lg border border-[#2d2d44] p-1">
            <button onClick={() => setMode('login')}
              className={`flex-1 py-2 text-sm rounded font-medium transition-colors ${mode === 'login' ? 'bg-[#4fc3f7]/20 text-[#4fc3f7]' : 'text-[#9e9eb0] hover:text-[#e0e0e0]'}`}>
              <KeyRound size={14} className="inline mr-1.5" />Login
            </button>
            <button onClick={() => setMode('register')}
              className={`flex-1 py-2 text-sm rounded font-medium transition-colors ${mode === 'register' ? 'bg-[#4fc3f7]/20 text-[#4fc3f7]' : 'text-[#9e9eb0] hover:text-[#e0e0e0]'}`}>
              <Shield size={14} className="inline mr-1.5" />Register
            </button>
          </div>

          {mode === 'login' ? (
            <form onSubmit={handleLogin} className="space-y-4">
              <div>
                <label className="block text-xs text-[#9e9eb0] mb-1.5">API Key</label>
                <div className="relative">
                  <input type={showKey ? 'text' : 'password'} value={apiKey} onChange={(e) => setApiKey(e.target.value)}
                    placeholder="Enter your API key..." autoFocus
                    className="w-full bg-[#0f0f1a] border border-[#2d2d44] rounded-lg pl-3 pr-20 py-2.5 text-sm text-[#e0e0e0] placeholder:text-[#9e9eb0] focus:outline-none focus:border-[#4fc3f7]" />
                  <div className="absolute right-2 top-1/2 -translate-y-1/2 flex gap-1">
                    <button type="button" onClick={() => setShowKey(!showKey)} className="p-1 text-[#9e9eb0] hover:text-[#e0e0e0]">
                      {showKey ? <EyeOff size={14} /> : <Eye size={14} />}
                    </button>
                  </div>
                </div>
                <button type="button" onClick={fillDevKey} className="text-[10px] text-[#4fc3f7] hover:underline mt-1">Use dev key</button>
              </div>
              {error && <p className="text-xs text-rose-400">{error}</p>}
              <button type="submit" disabled={loading || !apiKey.trim()}
                className="w-full aurelius-btn py-2.5 text-sm font-medium disabled:opacity-50">
                {loading ? <Loader2 size={16} className="animate-spin mx-auto" /> : 'Sign In'}
              </button>
            </form>
          ) : (
            <form onSubmit={handleRegister} className="space-y-4">
              <div>
                <label className="block text-xs text-[#9e9eb0] mb-1.5">Username</label>
                <input type="text" value={username} onChange={(e) => setUsername(e.target.value)}
                  placeholder="Choose a username..." autoFocus minLength={3} maxLength={32}
                  className="w-full bg-[#0f0f1a] border border-[#2d2d44] rounded-lg px-3 py-2.5 text-sm text-[#e0e0e0] placeholder:text-[#9e9eb0] focus:outline-none focus:border-[#4fc3f7]" />
              </div>
              <p className="text-[10px] text-[#9e9eb0]/60">An API key will be generated for your account.</p>
              <button type="submit" disabled={loading || !username.trim()}
                className="w-full aurelius-btn py-2.5 text-sm font-medium disabled:opacity-50">
                {loading ? <Loader2 size={16} className="animate-spin mx-auto" /> : 'Create Account'}
              </button>
            </form>
          )}
        </div>

        <p className="text-center text-[10px] text-[#9e9eb0]/40">Aurelius v0.1.0 &middot; Mission Control</p>
      </div>
    </div>
  )
}
