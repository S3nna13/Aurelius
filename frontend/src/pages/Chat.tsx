import { MessageSquare, Send, User, Bot } from 'lucide-react';
import { useState } from 'react';

const initialMessages = [
  { from: 'agent', text: 'Welcome to Aurelius Mission Control. How can I assist you today?' },
  { from: 'user', text: 'Show me the current system status.' },
  { from: 'agent', text: 'All systems nominal. 4 agents online, 127 tasks completed today, system health at 98%.' },
];

export default function Chat() {
  const [messages] = useState(initialMessages);

  return (
    <div className="h-full flex flex-col aurelius-card">
      <div className="flex items-center justify-between pb-4 border-b border-aurelius-border">
        <h2 className="text-sm font-semibold text-aurelius-text uppercase tracking-wider flex items-center gap-2">
          <MessageSquare size={16} className="text-aurelius-accent" />
          Agent Chat
        </h2>
        <span className="text-xs text-aurelius-muted">OpenClaw v1.0</span>
      </div>

      <div className="flex-1 overflow-y-auto py-4 space-y-4">
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex gap-3 ${msg.from === 'user' ? 'flex-row-reverse' : ''}`}
          >
            <div
              className={`w-8 h-8 rounded-full flex items-center justify-center shrink-0 ${
                msg.from === 'user'
                  ? 'bg-aurelius-accent/20 text-aurelius-accent'
                  : 'bg-aurelius-border/40 text-aurelius-muted'
              }`}
            >
              {msg.from === 'user' ? <User size={14} /> : <Bot size={14} />}
            </div>
            <div
              className={`max-w-[70%] px-4 py-2.5 rounded-xl text-sm leading-relaxed ${
                msg.from === 'user'
                  ? 'bg-aurelius-accent/10 text-aurelius-text border border-aurelius-accent/20'
                  : 'bg-aurelius-bg/60 text-aurelius-text border border-aurelius-border/50'
              }`}
            >
              {msg.text}
            </div>
          </div>
        ))}
      </div>

      <div className="pt-4 border-t border-aurelius-border">
        <div className="flex gap-2">
          <input
            type="text"
            placeholder="Type a message..."
            className="flex-1 bg-aurelius-bg border border-aurelius-border rounded-lg px-4 py-2 text-sm text-aurelius-text placeholder:text-aurelius-muted focus:outline-none focus:border-aurelius-accent"
          />
          <button className="aurelius-btn p-2.5">
            <Send size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}
