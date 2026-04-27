import { useState, useEffect, useCallback } from 'react'
import { Users as UsersIcon, Copy, Check, RefreshCw, Trash2, Loader2, Bot, Plus } from 'lucide-react'
import { useToast } from '../components/ToastProvider'
import { Modal } from '../components/ui/Modal'
import { Badge } from '../components/ui/Badge'

interface User {
  id: string
  username: string
  role: string
  apiKeys: number
  createdAt: string
}

interface ApiKey {
  prefix: string
  full?: string
}

export default function Users() {
  const { toast } = useToast()
  const [users, setUsers] = useState<User[]>([])
  const [keys, setKeys] = useState<ApiKey[]>([])
  const [loading, setLoading] = useState(true)
  const [generating, setGenerating] = useState(false)
  const [newKeyModal, setNewKeyModal] = useState(false)
  const [newKey, setNewKey] = useState('')
  const [copied, setCopied] = useState(false)

  const fetchData = useCallback(async () => {
    try {
      const [usersRes, keysRes] = await Promise.all([
        fetch('/api/auth/users'),
        fetch('/api/auth/keys'),
      ])
      if (usersRes.ok) setUsers((await usersRes.json()).users || [])
      if (keysRes.ok) setKeys((await keysRes.json()).keys || [])
    } catch { /* ignore */ }
    setLoading(false)
  }, [])

  useEffect(() => { fetchData() }, [fetchData])

  const generateKey = async () => {
    setGenerating(true)
    try {
      const res = await fetch('/api/auth/keys/generate', { method: 'POST' })
      const data = await res.json()
      if (data.success) {
        setNewKey(data.apiKey)
        setNewKeyModal(true)
        fetchData()
      }
    } catch { toast('Failed to generate key', 'error') }
    setGenerating(false)
  }

  const deleteKey = async (prefix: string) => {
    try {
      const res = await fetch(`/api/auth/keys/${prefix}`, { method: 'DELETE' })
      if (res.ok) {
        toast('API key deleted', 'info')
        fetchData()
      }
    } catch { toast('Failed to delete key', 'error') }
  }

  const copyKey = async () => {
    try {
      await navigator.clipboard.writeText(newKey)
      setCopied(true)
      setTimeout(() => setCopied(false), 2000)
      toast('API key copied', 'success')
    } catch { /* ignore */ }
  }

  return (
    <div className="space-y-6 max-w-4xl">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2"><UsersIcon size={20} className="text-[#4fc3f7]" /> Users</h2>
        <div className="flex gap-2">
          <button onClick={generateKey} disabled={generating} className="aurelius-btn flex items-center gap-1.5 text-sm disabled:opacity-50">
            {generating ? <Loader2 size={14} className="animate-spin" /> : <Plus size={14} />} Generate Key
          </button>
          <button onClick={fetchData} disabled={loading} className="aurelius-btn-outline flex items-center gap-1.5 text-sm disabled:opacity-50">
            <RefreshCw size={14} /> Refresh
          </button>
        </div>
      </div>

      {loading ? (
        <div className="aurelius-card text-center py-12 text-[#9e9eb0]"><Loader2 size={24} className="mx-auto mb-2 animate-spin opacity-60" /><p>Loading users...</p></div>
      ) : (
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          <div className="lg:col-span-2 aurelius-card space-y-4">
            <h3 className="text-sm font-semibold text-[#e0e0e0] uppercase tracking-wider">Registered Users</h3>
            {users.length === 0 ? <p className="text-sm text-[#9e9eb0] text-center py-6">No users yet.</p> : (
              <div className="space-y-2">
                {users.map((user) => (
                  <div key={user.id} className="flex items-center justify-between px-3 py-2.5 rounded-lg bg-[#0f0f1a]/50 border border-[#2d2d44]/50">
                    <div className="flex items-center gap-3">
                      <div className="w-8 h-8 rounded-full bg-[#4fc3f7]/10 text-[#4fc3f7] flex items-center justify-center"><Bot size={14} /></div>
                      <div>
                        <p className="text-sm font-medium text-[#e0e0e0]">{user.username}</p>
                        <p className="text-[10px] text-[#9e9eb0]">ID: {user.id} | {user.apiKeys} key(s)</p>
                      </div>
                    </div>
                    <Badge variant={user.role === 'admin' ? 'info' : 'default'}>{user.role}</Badge>
                  </div>
                ))}
              </div>
            )}
          </div>

          <div className="lg:col-span-1 aurelius-card space-y-4">
            <h3 className="text-sm font-semibold text-[#e0e0e0] uppercase tracking-wider">API Keys</h3>
            {keys.length === 0 ? <p className="text-sm text-[#9e9eb0] text-center py-6">No keys.</p> : (
              <div className="space-y-2">
                {keys.map((key) => (
                  <div key={key.prefix} className="flex items-center justify-between px-3 py-2 rounded-lg bg-[#0f0f1a]/50 border border-[#2d2d44]/50 group">
                    <span className="font-mono text-xs text-[#e0e0e0]">{key.prefix}</span>
                    <button onClick={() => deleteKey(key.prefix)} className="opacity-0 group-hover:opacity-100 p-1 text-[#9e9eb0] hover:text-rose-400 transition-all"><Trash2 size={12} /></button>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      )}

      <Modal open={newKeyModal} onClose={() => { setNewKeyModal(false); setNewKey('') }} title="New API Key Generated" size="md">
        <div className="space-y-4">
          <p className="text-sm text-[#9e9eb0]">Save this key - it won't be shown again.</p>
          <div className="flex items-center gap-2 p-3 bg-[#0f0f1a] border border-[#2d2d44] rounded-lg">
            <code className="flex-1 text-sm font-mono text-[#e0e0e0] break-all">{newKey}</code>
            <button onClick={copyKey} className="p-2 rounded text-[#9e9eb0] hover:text-[#4fc3f7] hover:bg-[#4fc3f7]/10 transition-colors">
              {copied ? <Check size={16} className="text-emerald-400" /> : <Copy size={16} />}
            </button>
          </div>
          <button onClick={() => { setNewKeyModal(false); setNewKey('') }} className="w-full aurelius-btn py-2 text-sm">Done</button>
        </div>
      </Modal>
    </div>
  )
}
