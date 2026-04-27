import { useState, useEffect, useCallback, useRef } from 'react'
import { Send, Sliders, Loader2, Bot, User, Trash2, Wifi, WifiOff } from 'lucide-react'
import { api } from '../api/AureliusClient'
import { useWebSocket } from '../hooks/useWebSocket'
import type { ModelInfo } from '../api/types'

interface ChatMsg {
  id: string
  role: 'user' | 'assistant'
  content: string
}

export default function Playground() {
  const [models, setModels] = useState<ModelInfo[]>([])
  const [selectedModel, setSelectedModel] = useState('')
  const [prompt, setPrompt] = useState('')
  const [messages, setMessages] = useState<ChatMsg[]>([])
  const [sending, setSending] = useState(false)
  const [temperature, setTemperature] = useState(0.7)
  const [maxTokens, setMaxTokens] = useState(512)
  const [topP, setTopP] = useState(1.0)
  const [showParams, setShowParams] = useState(false)
  const [, setBackendStatus] = useState<'unknown' | 'online' | 'offline'>('unknown')
  const scrollRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)
  const { connected, send, on, off } = useWebSocket()
  const bufferRef = useRef('')
  const pendingIdRef = useRef<string | null>(null)

  useEffect(() => {
    api.listModels().then((res) => {
      setModels(res.models || [])
      if (res.models?.length && !selectedModel) setSelectedModel(res.models[0].id)
    }).catch(() => {})
  }, [])

  useEffect(() => {
    if (!selectedModel) return
    api.getModel(selectedModel).then((m) => {
      setBackendStatus(m.state === 'loaded' ? 'online' : 'offline')
    }).catch(() => setBackendStatus('offline'))
  }, [selectedModel])

  useEffect(() => {
    if (scrollRef.current) scrollRef.current.scrollTop = scrollRef.current.scrollHeight
  }, [messages])

  // Handle streaming tokens via WebSocket
  useEffect(() => {
    const onToken = (payload: unknown) => {
      const d = payload as { token?: string }
      if (d.token) {
        bufferRef.current += d.token
        setMessages((prev) => prev.map((m) =>
          m.id === pendingIdRef.current ? { ...m, content: bufferRef.current } : m
        ))
      }
    }

    const onDone = () => {
      pendingIdRef.current = null
      bufferRef.current = ''
      setSending(false)
    }

    const onError = (payload: unknown) => {
      const d = payload as { message?: string }
      setMessages((prev) => prev.map((m) =>
        m.id === pendingIdRef.current ? { ...m, content: `Error: ${d.message || 'Unknown error'}` } : m
      ))
      pendingIdRef.current = null
      bufferRef.current = ''
      setSending(false)
    }

    on('chat_token', onToken)
    on('chat_done', onDone)
    on('chat_error', onError)
    return () => { off('chat_token', onToken); off('chat_done', onDone); off('chat_error', onError) }
  }, [on, off])

  const genId = () => Math.random().toString(36).slice(2)

  const sendMessage = useCallback(async () => {
    if (!prompt.trim() || sending) return
    const text = prompt.trim()
    const userMsg: ChatMsg = { id: genId(), role: 'user', content: text }
    const pendingId = genId()
    pendingIdRef.current = pendingId
    bufferRef.current = ''
    const pendingMsg: ChatMsg = { id: pendingId, role: 'assistant', content: '' }
    setMessages((prev) => [...prev, userMsg, pendingMsg])
    setPrompt('')
    setSending(true)
    setBackendStatus('online')

    if (connected) {
      send('chat', { text })
    } else {
      try {
        const res = await api.postCommand(text)
        const content = res.output || res.error || '(empty)'
        setMessages((prev) => prev.map((m) => m.id === pendingId ? { ...m, content } : m))
        if (!res.success) setBackendStatus('offline')
      } catch (err) {
        setMessages((prev) => prev.map((m) => m.id === pendingId ? { ...m, content: `Error: ${err instanceof Error ? err.message : 'Failed'}` } : m))
        setBackendStatus('offline')
      } finally {
        pendingIdRef.current = null
        setSending(false)
        inputRef.current?.focus()
      }
    }
  }, [prompt, sending, connected, send])

  const clearChat = () => { setMessages([]); setPrompt('') }
  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendMessage() }
  }

  return (
    <div className="h-full flex flex-col">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
          <Bot size={20} className="text-[#4fc3f7]" /> Inference Playground
          {connected && <span className="ml-2 text-[10px] font-bold text-emerald-400 bg-emerald-500/10 px-1.5 py-0.5 rounded border border-emerald-500/20"><Wifi size={10} className="inline mr-0.5" />Streaming</span>}
        </h2>
        <div className="flex items-center gap-3">
          <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full border inline-flex items-center gap-1 ${
            connected ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
            : 'bg-[#2d2d44]/20 text-[#9e9eb0] border-[#2d2d44]/40'
          }`}>
            {connected ? <Wifi size={10} /> : <WifiOff size={10} />}
            {connected ? 'Streaming' : 'REST'}
          </span>
          <button onClick={clearChat} className="text-[#9e9eb0] hover:text-rose-400 transition-colors" title="Clear"><Trash2 size={14} /></button>
        </div>
      </div>

      <div className="flex gap-3 mb-4 flex-wrap">
        <div className="flex-1 min-w-[200px]">
          <label className="block text-[10px] text-[#9e9eb0] uppercase tracking-wider mb-1">Model</label>
          <select value={selectedModel} onChange={(e) => setSelectedModel(e.target.value)}
            className="w-full bg-[#0f0f1a] border border-[#2d2d44] rounded-lg px-3 py-2 text-sm text-[#e0e0e0] focus:outline-none focus:border-[#4fc3f7]">
            {models.map((m) => <option key={m.id} value={m.id}>{m.name}</option>)}
          </select>
        </div>
        <div className="self-end">
          <button onClick={() => setShowParams(!showParams)}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium border transition-colors ${
              showParams ? 'bg-[#4fc3f7]/10 text-[#4fc3f7] border-[#4fc3f7]/30' : 'bg-[#0f0f1a] text-[#9e9eb0] border-[#2d2d44] hover:border-[#4fc3f7]/30'
            }`}>
            <Sliders size={14} /> Params
          </button>
        </div>
      </div>

      {showParams && (
        <div className="aurelius-card grid grid-cols-3 gap-4 mb-4">
          <div>
            <label className="block text-[10px] text-[#9e9eb0] mb-1">Temperature: {temperature.toFixed(1)}</label>
            <input type="range" min="0" max="2" step="0.1" value={temperature} onChange={(e) => setTemperature(parseFloat(e.target.value))}
              className="w-full accent-[#4fc3f7]" />
          </div>
          <div>
            <label className="block text-[10px] text-[#9e9eb0] mb-1">Max Tokens: {maxTokens}</label>
            <input type="range" min="64" max="4096" step="64" value={maxTokens} onChange={(e) => setMaxTokens(parseInt(e.target.value))}
              className="w-full accent-[#4fc3f7]" />
          </div>
          <div>
            <label className="block text-[10px] text-[#9e9eb0] mb-1">Top-P: {topP.toFixed(1)}</label>
            <input type="range" min="0.1" max="1" step="0.1" value={topP} onChange={(e) => setTopP(parseFloat(e.target.value))}
              className="w-full accent-[#4fc3f7]" />
          </div>
        </div>
      )}

      <div className="flex-1 overflow-y-auto space-y-3 min-h-0 mb-4" ref={scrollRef}>
        {messages.length === 0 && (
          <div className="text-center py-16 text-[#9e9eb0]">
            <Bot size={40} className="mx-auto mb-4 opacity-30" />
            <p className="text-sm">Send a prompt to test the model.</p>
            <p className="text-xs mt-1 opacity-60">{connected ? 'Responses stream in real-time via WebSocket.' : 'Using REST API (offline mode).'}</p>
          </div>
        )}
        {messages.map((msg) => (
          <div key={msg.id} className={`flex gap-3 ${msg.role === 'user' ? 'flex-row-reverse' : ''}`}>
            <div className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${
              msg.role === 'user' ? 'bg-[#4fc3f7]/20 text-[#4fc3f7]' : 'bg-[#2d2d44]/40 text-[#9e9eb0]'
            }`}>
              {msg.role === 'user' ? <User size={14} /> : <Bot size={14} />}
            </div>
            <div className={`max-w-[75%] px-4 py-2.5 rounded-xl text-sm leading-relaxed whitespace-pre-wrap ${
              msg.role === 'user' ? 'bg-[#4fc3f7]/10 text-[#e0e0e0] border border-[#4fc3f7]/20'
              : 'bg-[#0f0f1a]/60 text-[#e0e0e0] border border-[#2d2d44]/50'
            }`}>
              {msg.content || (msg.id === pendingIdRef.current ? <span className="text-[#4fc3f7] animate-pulse">▊</span> : '')}
            </div>
          </div>
        ))}
      </div>

      <div className="flex gap-2 items-end">
        <textarea ref={inputRef} value={prompt} onChange={(e) => setPrompt(e.target.value)} onKeyDown={handleKeyDown}
          placeholder="Enter your prompt... (Shift+Enter for newline)" rows={2}
          className="flex-1 bg-[#0f0f1a] border border-[#2d2d44] rounded-lg px-4 py-2.5 text-sm text-[#e0e0e0] placeholder:text-[#9e9eb0] focus:outline-none focus:border-[#4fc3f7] resize-none disabled:opacity-50" />
        <button onClick={sendMessage} disabled={sending || !prompt.trim()} className="aurelius-btn p-3 disabled:opacity-50">
          {sending ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
        </button>
      </div>
    </div>
  )
}
