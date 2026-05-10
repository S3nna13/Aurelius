import { useState, useRef, useEffect } from 'react';
import { motion } from 'framer-motion';
import {
  Play, Square, RotateCcw, Terminal, MessageSquare,
  Bot, Cpu, Clock, Loader2, Save, Trash2, Code,
} from 'lucide-react';
import Input from '../components/ui/Input';
import Select from '../components/ui/Select';
import Toggle from '../components/ui/Toggle';

interface TestResult {
  type: 'thought' | 'tool_call' | 'tool_result' | 'error' | 'output';
  content: string;
  timestamp: number;
  duration?: number;
}

export default function AgentPlayground() {
  const [prompt, setPrompt] = useState('');
  const [agentType, setAgentType] = useState('coding');
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(512);
  const [results, setResults] = useState<TestResult[]>([]);
  const [running, setRunning] = useState(false);
  const [streamingText, setStreamingText] = useState('');
  const [savedPrompts, setSavedPrompts] = useState<string[]>([]);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [results, streamingText]);

  const handleRun = async () => {
    if (!prompt.trim()) return;
    setRunning(true);
    setStreamingText('');
    setResults(prev => [...prev, { type: 'thought', content: `Routing to ${agentType} agent...`, timestamp: Date.now() }]);

    const startTime = Date.now();
    try {
      const res = await fetch('/api/chat/agent', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: prompt, stream: true }),
      });

      if (!res.ok) throw new Error('Request failed');
      const reader = res.body?.getReader();
      const decoder = new TextDecoder();
      let text = '';

      if (reader) {
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
          const chunk = decoder.decode(value);
          for (const line of chunk.split('\n')) {
            if (line.startsWith('data: ')) {
              try {
                const data = JSON.parse(line.slice(6));
                if (data.content) { text += data.content; setStreamingText(text); }
                if (data.agent) setResults(prev => [...prev, { type: 'tool_call', content: `Agent: ${data.agent}`, timestamp: Date.now() }]);
              } catch {}
            }
          }
        }
      }

      setResults(prev => [...prev, {
        type: 'output', content: text, timestamp: Date.now(), duration: Date.now() - startTime,
      }]);
    } catch (e: any) {
      setResults(prev => [...prev, { type: 'error', content: e.message, timestamp: Date.now() }]);
    }
    setStreamingText('');
    setRunning(false);
  };

  const savePrompt = () => {
    if (prompt.trim()) {
      setSavedPrompts(prev => [prompt, ...prev].slice(0, 10));
    }
  };

  const clearResults = () => setResults([]);

  return (
    <div className="flex gap-4 h-[calc(100vh-8rem)]">
      <div className="flex-1 flex flex-col space-y-4">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2 shrink-0">
          <Bot size={20} className="text-[#4fc3f7]" /> Agent Playground
        </h2>

        <div className="aurelius-card p-4 flex-1 flex flex-col">
          <div className="flex-1 overflow-y-auto space-y-2 mb-4">
            {results.length === 0 && !running && (
              <div className="text-center py-16 text-[#9e9eb0]">
                <Terminal size={32} className="mx-auto mb-3 opacity-40" />
                <p className="text-sm">Enter a prompt and run to test agent behavior.</p>
              </div>
            )}
            {results.map((r, i) => (
              <motion.div key={i} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
                className={`p-3 rounded-lg border text-sm ${
                  r.type === 'error' ? 'border-rose-500/20 bg-rose-500/5' :
                  r.type === 'tool_call' ? 'border-amber-500/20 bg-amber-500/5' :
                  r.type === 'output' ? 'border-emerald-500/20 bg-emerald-500/5' :
                  'border-[#2d2d44] bg-[#0f0f1a]'
                }`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span className={`text-[10px] font-bold uppercase ${
                    r.type === 'error' ? 'text-rose-400' :
                    r.type === 'tool_call' ? 'text-amber-400' :
                    r.type === 'output' ? 'text-emerald-400' : 'text-[#4fc3f7]'
                  }`}>{r.type}</span>
                  {r.duration && <span className="text-[10px] text-[#9e9eb0]">({r.duration}ms)</span>}
                </div>
                <p className="text-xs text-[#9e9eb0] whitespace-pre-wrap">{r.content}</p>
              </motion.div>
            ))}
            {running && streamingText && (
              <div className="p-3 rounded-lg border border-[#4fc3f7]/20 bg-[#4fc3f7]/5">
                <p className="text-xs text-[#9e9eb0]">{streamingText}<span className="animate-pulse">▊</span></p>
              </div>
            )}
            <div ref={bottomRef} />
          </div>

          <div className="space-y-3">
            <div className="flex gap-2">
              <textarea value={prompt} onChange={e => setPrompt(e.target.value)}
                placeholder="Enter a prompt to test..." rows={2}
                className="flex-1 bg-[#0f0f1a] border border-[#2d2d44] rounded-lg p-3 text-sm text-[#e0e0e0] placeholder:text-[#9e9eb0] resize-none" />
            </div>
            <div className="flex items-center gap-2">
              <button onClick={handleRun} disabled={running || !prompt.trim()}
                className="aurelius-btn-primary flex items-center gap-2 text-sm disabled:opacity-50">
                {running ? <Loader2 size={14} className="animate-spin" /> : <Play size={14} />}
                {running ? 'Running...' : 'Run'}
              </button>
              <button onClick={clearResults} className="aurelius-btn-outline p-2"><Trash2 size={14} /></button>
              <button onClick={savePrompt} className="aurelius-btn-outline p-2"><Save size={14} /></button>
              <div className="flex-1" />
              <span className="text-[10px] text-[#9e9eb0]">{agentType} · T={temperature} · max={maxTokens}</span>
            </div>
          </div>
        </div>
      </div>

      <div className="w-56 hidden lg:flex flex-col space-y-4">
        <div className="aurelius-card p-4 space-y-3">
          <h3 className="text-xs font-bold text-[#e0e0e0] uppercase tracking-wider">Config</h3>
          <div>
            <label className="text-[10px] text-[#9e9eb0]">Agent Type</label>
            <select value={agentType} onChange={e => setAgentType(e.target.value)}
              className="w-full bg-[#0f0f1a] border border-[#2d2d44] rounded-lg px-2 py-1.5 text-xs text-[#e0e0e0] mt-1">
              <option value="coding">Coding</option>
              <option value="research">Research</option>
              <option value="general">General</option>
            </select>
          </div>
          <div>
            <label className="text-[10px] text-[#9e9eb0]">Temperature</label>
            <input type="range" min={0} max={2} step={0.05} value={temperature}
              onChange={e => setTemperature(parseFloat(e.target.value))}
              className="w-full accent-[#4fc3f7] mt-1" />
          </div>
          <div>
            <label className="text-[10px] text-[#9e9eb0]">Max Tokens</label>
            <input type="range" min={64} max={4096} step={64} value={maxTokens}
              onChange={e => setMaxTokens(parseInt(e.target.value))}
              className="w-full accent-[#4fc3f7] mt-1" />
          </div>
        </div>

        {savedPrompts.length > 0 && (
          <div className="aurelius-card p-4 space-y-2">
            <h3 className="text-xs font-bold text-[#e0e0e0] uppercase tracking-wider">Saved Prompts</h3>
            {savedPrompts.map((p, i) => (
              <button key={i} onClick={() => setPrompt(p)}
                className="w-full text-left text-[10px] text-[#9e9eb0] truncate hover:text-[#e0e0e0] p-1 rounded">
                {p.slice(0, 60)}...
              </button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
