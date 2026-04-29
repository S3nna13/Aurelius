import { useState, useRef, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Send, Bot, User, Cpu, MessageSquare, Terminal,
  Loader2, RefreshCw, Sparkles, History, Plus,
} from 'lucide-react';

interface ChatMessage {
  role: 'user' | 'assistant' | 'system';
  content: string;
  agent?: string;
}

interface Conversation {
  id: string; messageCount: number; agent: string; lastMessage: string;
}

export default function Chat() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [currentAgent, setCurrentAgent] = useState<string | null>(null);
  const [streamingText, setStreamingText] = useState('');
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [convId, setConvId] = useState<string | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);
  const eventSourceRef = useRef<EventSource | null>(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [messages, streamingText]);

  const handleSubmit = useCallback(async () => {
    if (!input.trim() || loading) return;
    const userMessage = input.trim();
    setInput('');
    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    setLoading(true);
    setStreamingText('');

    try {
      const res = await fetch('/api/chat/agent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          message: userMessage,
          conversationId: convId,
          stream: true,
        }),
      });

      if (!res.ok) throw new Error('Request failed');
      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      let fullText = '';
      let detectedAgent: string | null = null;

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value);
          for (const line of chunk.split('\n')) {
            if (line.startsWith('event: ')) {
              const eventType = line.slice(7);
              if (eventType === 'agent') detectedAgent = eventType;
              continue;
            }
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.agent) {
                  detectedAgent = data.agent;
                  setCurrentAgent(data.agent);
                }
                if (data.content) {
                  fullText += data.content;
                  setStreamingText(fullText);
                }
              } catch { /* ignore */ }
            }
          }
        }
      }

      setMessages(prev => [...prev, { role: 'assistant', content: fullText, agent: detectedAgent || undefined }]);
    } catch (e: any) {
      setMessages(prev => [...prev, { role: 'assistant', content: `Error: ${e.message}` }]);
    }
    setStreamingText('');
    setLoading(false);
  }, [input, loading, convId]);

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); handleSubmit(); }
  };

  const newConversation = () => {
    setMessages([]);
    setConvId(null);
    setCurrentAgent(null);
  };

  return (
    <div className="flex gap-4 h-[calc(100vh-8rem)]">
      {/* Conversation sidebar */}
      <div className="w-64 hidden lg:flex flex-col gap-2">
        <button onClick={newConversation} className="aurelius-btn-primary flex items-center gap-2 text-sm mb-2">
          <Plus size={14} /> New Chat
        </button>
        <div className="flex-1 overflow-y-auto space-y-1">
          {conversations.map(conv => (
            <div key={conv.id} className="aurelius-card p-2 text-xs cursor-pointer hover:border-[#4fc3f7]/30">
              <p className="text-[#e0e0e0] truncate">{conv.lastMessage}</p>
              <p className="text-[#9e9eb0] mt-0.5">{conv.agent} · {conv.messageCount} msgs</p>
            </div>
          ))}
        </div>
      </div>

      {/* Main chat area */}
      <div className="flex-1 flex flex-col aurelius-card">
        <div className="flex items-center justify-between px-4 py-3 border-b border-[#2d2d44]">
          <div className="flex items-center gap-2">
            <MessageSquare size={16} className="text-[#4fc3f7]" />
            <span className="text-sm font-medium text-[#e0e0e0]">Agent Chat</span>
            {currentAgent && (
              <span className="text-[10px] text-[#4fc3f7] bg-[#4fc3f7]/10 px-2 py-0.5 rounded-full border border-[#4fc3f7]/20">
                {currentAgent}
              </span>
            )}
          </div>
        </div>

        <div className="flex-1 overflow-y-auto p-4 space-y-4">
          {messages.length === 0 && !loading && (
            <div className="text-center py-16">
              <Bot size={40} className="mx-auto text-[#4fc3f7] mb-3 opacity-50" />
              <p className="text-[#9e9eb0] text-sm">Ask me anything — I'll route it to the best agent.</p>
              <p className="text-[#9e9eb0] text-xs mt-1">Try "write a function" for the Coding Agent, or "research a topic" for the Research Agent.</p>
            </div>
          )}

          <AnimatePresence>
            {messages.map((msg, i) => (
              <motion.div key={i} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : ''}`}
              >
                {msg.role !== 'user' && (
                  <div className="w-8 h-8 rounded-lg bg-[#4fc3f7]/10 flex items-center justify-center shrink-0">
                    <Bot size={14} className="text-[#4fc3f7]" />
                  </div>
                )}
                <div className={`max-w-[80%] ${msg.role === 'user' ? 'order-first' : ''}`}>
                  {msg.role === 'user' ? (
                    <div className="bg-[#4fc3f7]/10 border border-[#4fc3f7]/20 rounded-2xl rounded-tr-md px-4 py-2.5">
                      <p className="text-sm text-[#e0e0e0] whitespace-pre-wrap">{msg.content}</p>
                    </div>
                  ) : (
                    <div className="bg-[#0f0f1a] border border-[#2d2d44] rounded-2xl rounded-tl-md px-4 py-2.5">
                      {msg.agent && (
                        <span className="text-[10px] text-[#4fc3f7] font-bold uppercase tracking-wider block mb-1">
                          <Cpu size={10} className="inline mr-1" />{msg.agent}
                        </span>
                      )}
                      <p className="text-sm text-[#e0e0e0] whitespace-pre-wrap">{msg.content}</p>
                    </div>
                  )}
                </div>
                {msg.role === 'user' && (
                  <div className="w-8 h-8 rounded-lg bg-amber-500/10 flex items-center justify-center shrink-0">
                    <User size={14} className="text-amber-400" />
                  </div>
                )}
              </motion.div>
            ))}

            {loading && streamingText && (
              <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} className="flex gap-3">
                <div className="w-8 h-8 rounded-lg bg-[#4fc3f7]/10 flex items-center justify-center shrink-0">
                  <Bot size={14} className="text-[#4fc3f7]" />
                </div>
                <div className="bg-[#0f0f1a] border border-[#2d2d44] rounded-2xl rounded-tl-md px-4 py-2.5">
                  {currentAgent && (
                    <span className="text-[10px] text-[#4fc3f7] font-bold uppercase tracking-wider block mb-1">
                      <Cpu size={10} className="inline mr-1" />{currentAgent}
                    </span>
                  )}
                  <p className="text-sm text-[#e0e0e0]">{streamingText}<span className="animate-pulse">▊</span></p>
                </div>
              </motion.div>
            )}
          </AnimatePresence>
          <div ref={bottomRef} />
        </div>

        <div className="border-t border-[#2d2d44] p-4">
          <div className="flex gap-2">
            <input
              value={input}
              onChange={e => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask the agent network..."
              disabled={loading}
              className="flex-1 bg-[#0f0f1a] border border-[#2d2d44] rounded-xl px-4 py-2.5 text-sm text-[#e0e0e0] placeholder:text-[#9e9eb0] focus:outline-none focus:border-[#4fc3f7]/50 disabled:opacity-50"
            />
            <button
              onClick={handleSubmit}
              disabled={!input.trim() || loading}
              className="aurelius-btn-primary px-4 rounded-xl disabled:opacity-50"
            >
              {loading ? <Loader2 size={16} className="animate-spin" /> : <Send size={16} />}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
