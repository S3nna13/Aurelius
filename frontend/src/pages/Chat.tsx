import { useState, useRef, useEffect, useCallback } from 'react';
import {
  MessageSquare,
  Send,
  User,
  Bot,
  Loader2,
  Trash2,
  Zap,
  Activity,
  Wrench,
  GitBranch,
  AlertTriangle,
} from 'lucide-react';
import { useToast } from '../components/ToastProvider';

interface ChatMessage {
  id: string;
  from: 'user' | 'agent' | 'system';
  text: string;
  timestamp: number;
  pending?: boolean;
}

const STORAGE_KEY = 'aurelius-chat-history';
const MAX_HISTORY = 100;

const quickActions = [
  { label: 'System Status', icon: Activity, text: 'What is the current system status?' },
  { label: 'List Skills', icon: Wrench, text: 'List all active skills' },
  { label: 'Run Health Check', icon: Zap, text: 'Run a health check' },
  { label: 'Workflow Status', icon: GitBranch, text: 'Show workflow status' },
];

function generateId(): string {
  return Math.random().toString(36).slice(2) + Date.now().toString(36);
}

function loadHistory(): ChatMessage[] {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (raw) {
      const parsed = JSON.parse(raw);
      if (Array.isArray(parsed)) return parsed;
    }
  } catch {
    // ignore
  }
  return [
    {
      id: generateId(),
      from: 'agent',
      text: 'Welcome to Aurelius Mission Control. How can I assist you today?',
      timestamp: Date.now(),
    },
  ];
}

function saveHistory(messages: ChatMessage[]) {
  try {
    const trimmed = messages.slice(-MAX_HISTORY);
    localStorage.setItem(STORAGE_KEY, JSON.stringify(trimmed));
  } catch {
    // ignore
  }
}

export default function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>(loadHistory);
  const [input, setInput] = useState('');
  const [sending, setSending] = useState(false);
  const [sseConnected, setSseConnected] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const { toast } = useToast();

  // Auto-scroll to bottom
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  // Persist to localStorage
  useEffect(() => {
    saveHistory(messages);
  }, [messages]);

  // SSE for real-time system events
  useEffect(() => {
    const es = new EventSource('/api/events');
    es.onopen = () => setSseConnected(true);
    es.onerror = () => setSseConnected(false);
    es.addEventListener('notification', (e) => {
      try {
        const data = JSON.parse(e.data);
        setMessages((prev) => [
          ...prev,
          {
            id: generateId(),
            from: 'system',
            text: `**${data.title}**\n${data.body}`,
            timestamp: Date.now(),
          },
        ]);
      } catch {
        // ignore parse errors
      }
    });
    return () => es.close();
  }, []);

  const sendMessage = useCallback(
    async (text: string) => {
      if (!text.trim() || sending) return;
      const userMsg: ChatMessage = {
        id: generateId(),
        from: 'user',
        text: text.trim(),
        timestamp: Date.now(),
      };
      const pendingMsg: ChatMessage = {
        id: generateId(),
        from: 'agent',
        text: '',
        timestamp: Date.now(),
        pending: true,
      };
      setMessages((prev) => [...prev, userMsg, pendingMsg]);
      setInput('');
      setSending(true);

      try {
        const res = await fetch('/api/command', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ command: text.trim() }),
        });
        const data = await res.json();
        setMessages((prev) => {
          const filtered = prev.filter((m) => m.id !== pendingMsg.id);
          const response: ChatMessage = {
            id: generateId(),
            from: data.success ? 'agent' : 'system',
            text: data.output || data.error || 'No response',
            timestamp: Date.now(),
          };
          return [...filtered, response];
        });
        if (!data.success) {
          toast(data.error || 'Command failed', 'error');
        }
      } catch (err) {
        setMessages((prev) => {
          const filtered = prev.filter((m) => m.id !== pendingMsg.id);
          return [
            ...filtered,
            {
              id: generateId(),
              from: 'system',
              text: err instanceof Error ? err.message : 'Network error',
              timestamp: Date.now(),
            },
          ];
        });
        toast('Failed to send message', 'error');
      } finally {
        setSending(false);
        inputRef.current?.focus();
      }
    },
    [sending, toast]
  );

  // Listen for command palette dispatch
  useEffect(() => {
    const handler = (e: CustomEvent<string>) => {
      sendMessage(e.detail);
    };
    window.addEventListener('aurelius:send-command', handler as EventListener);
    return () => window.removeEventListener('aurelius:send-command', handler as EventListener);
  }, [sendMessage]);

  const clearHistory = () => {
    setMessages([
      {
        id: generateId(),
        from: 'agent',
        text: 'Chat history cleared. How can I assist you?',
        timestamp: Date.now(),
      },
    ]);
    toast('Chat history cleared', 'info');
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input);
    }
  };

  return (
    <div className="h-full flex flex-col aurelius-card">
      {/* Header */}
      <div className="flex items-center justify-between pb-4 border-b border-[#2d2d44]">
        <h2 className="text-sm font-semibold text-[#e0e0e0] uppercase tracking-wider flex items-center gap-2">
          <MessageSquare size={16} className="text-[#4fc3f7]" />
          Agent Chat
        </h2>
        <div className="flex items-center gap-3">
          <span
            className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${
              sseConnected
                ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                : 'bg-[#2d2d44]/20 text-[#9e9eb0] border-[#2d2d44]/40'
            }`}
          >
            {sseConnected ? 'Live' : 'Offline'}
          </span>
          <button
            onClick={clearHistory}
            className="text-[#9e9eb0] hover:text-rose-400 transition-colors"
            title="Clear chat"
          >
            <Trash2 size={14} />
          </button>
          <span className="text-xs text-[#9e9eb0]">OpenClaw v1.0</span>
        </div>
      </div>

      {/* Messages */}
      <div ref={scrollRef} className="flex-1 overflow-y-auto py-4 space-y-4 min-h-0">
        {messages.map((msg) => (
          <div
            key={msg.id}
            className={`flex gap-3 ${msg.from === 'user' ? 'flex-row-reverse' : ''}`}
          >
            <div
              className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${
                msg.from === 'user'
                  ? 'bg-[#4fc3f7]/20 text-[#4fc3f7]'
                  : msg.from === 'system'
                  ? 'bg-amber-500/10 text-amber-400'
                  : 'bg-[#2d2d44]/40 text-[#9e9eb0]'
              }`}
            >
              {msg.from === 'user' ? (
                <User size={14} />
              ) : msg.from === 'system' ? (
                <AlertTriangle size={14} />
              ) : (
                <Bot size={14} />
              )}
            </div>
            <div
              className={`max-w-[75%] px-4 py-2.5 rounded-xl text-sm leading-relaxed whitespace-pre-wrap ${
                msg.from === 'user'
                  ? 'bg-[#4fc3f7]/10 text-[#e0e0e0] border border-[#4fc3f7]/20'
                  : msg.from === 'system'
                  ? 'bg-amber-500/5 text-amber-300 border border-amber-500/20'
                  : 'bg-[#0f0f1a]/60 text-[#e0e0e0] border border-[#2d2d44]/50'
              }`}
            >
              {msg.pending ? (
                <div className="flex items-center gap-2 text-[#9e9eb0]">
                  <Loader2 size={14} className="animate-spin" />
                  <span>Thinking...</span>
                </div>
              ) : (
                msg.text
              )}
            </div>
          </div>
        ))}
      </div>

      {/* Quick Actions */}
      <div className="flex gap-2 overflow-x-auto pb-2">
        {quickActions.map((action) => (
          <button
            key={action.label}
            onClick={() => sendMessage(action.text)}
            disabled={sending}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-[#0f0f1a] border border-[#2d2d44] text-xs text-[#9e9eb0] hover:text-[#4fc3f7] hover:border-[#4fc3f7]/30 transition-colors whitespace-nowrap disabled:opacity-50"
          >
            <action.icon size={12} />
            {action.label}
          </button>
        ))}
      </div>

      {/* Input */}
      <div className="pt-3 border-t border-[#2d2d44]">
        <div className="flex gap-2">
          <input
            ref={inputRef}
            type="text"
            placeholder="Type a command or question..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyDown}
            disabled={sending}
            className="flex-1 bg-[#0f0f1a] border border-[#2d2d44] rounded-lg px-4 py-2.5 text-sm text-[#e0e0e0] placeholder:text-[#9e9eb0] focus:outline-none focus:border-[#4fc3f7] disabled:opacity-50"
          />
          <button
            onClick={() => sendMessage(input)}
            disabled={sending || !input.trim()}
            className="aurelius-btn p-2.5 disabled:opacity-50"
          >
            {sending ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
          </button>
        </div>
      </div>
    </div>
  );
}
