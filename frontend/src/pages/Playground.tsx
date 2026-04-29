import { useState, useRef, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  Play, Square, Settings, Trash2, Copy, Bot,
  Cpu, Thermometer, Sliders, Hash, Maximize2, Minimize2,
} from 'lucide-react';
import Select from '../components/ui/Select';
import Input from '../components/ui/Input';
import Textarea from '../components/ui/Textarea';
import Toggle from '../components/ui/Toggle';
import { useToast } from '../components/ToastProvider';
import { useLocalStorage } from '../hooks/useLocalStorage';

interface ModelConfig {
  id: string; name: string; provider: string;
}

interface ChatMessage {
  role: 'system' | 'user';
  content: string;
}

interface CompletionRequest {
  model: string;
  messages: ChatMessage[];
  temperature: number;
  max_tokens: number;
  top_p: number;
  stream: boolean;
}

interface CompletionResult {
  model: string; text: string; tokens: number;
  duration: number; timestamp: number;
}

const AVAILABLE_MODELS: ModelConfig[] = [
  { id: 'aurelius-1.3b', name: 'Aurelius 1.3B', provider: 'local' },
  { id: 'aurelius-2.7b', name: 'Aurelius 2.7B', provider: 'local' },
  { id: 'aurelius-3b', name: 'Aurelius 3.0B', provider: 'local' },
  { id: 'aurelius-moe', name: 'Aurelius MoE 5B', provider: 'local' },
];

export default function Playground() {
  const { toast } = useToast();
  const [prompt, setPrompt] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('You are a helpful assistant.');
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(512);
  const [topP, setTopP] = useState(0.9);
  const [stream, setStream] = useState(true);
  const [selectedModels, setSelectedModels] = useState<string[]>([AVAILABLE_MODELS[0].id]);
  const [results, setResults] = useLocalStorage<CompletionResult[]>('playground_results', []);
  const [running, setRunning] = useState(false);
  const [showSettings, setShowSettings] = useState(true);
  const [activeStreams, setActiveStreams] = useState<Record<string, string>>({});
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [results, activeStreams]);

  const toggleModel = (id: string) => {
    setSelectedModels(prev =>
      prev.includes(id) ? prev.filter(m => m !== id) : [...prev, id]
    );
  };

  const handleSubmit = useCallback(async () => {
    if (!prompt.trim() || selectedModels.length === 0) return;
    setRunning(true);
    setActiveStreams({});

    for (const modelId of selectedModels) {
      const startTime = Date.now();
      let fullText = '';
      const messages: ChatMessage[] = [
        ...(systemPrompt.trim() ? [{ role: 'system' as const, content: systemPrompt.trim() }] : []),
        { role: 'user', content: prompt.trim() },
      ];

      try {
        const payload: CompletionRequest = {
          model: modelId,
          messages,
          max_tokens: maxTokens,
          temperature,
          top_p: topP,
          stream,
        };

        const res = await fetch('/api/chat/completions', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(payload),
        });

        if (!res.ok) {
          throw new Error(`HTTP ${res.status}: ${res.statusText}`);
        }

        if (stream) {
          const reader = res.body?.getReader();
          const decoder = new TextDecoder();
          if (reader) {
            while (true) {
              const { done, value } = await reader.read();
              if (done) break;
              const chunk = decoder.decode(value);
              for (const line of chunk.split('\n')) {
                if (line.startsWith('data: ') && line !== 'data: [DONE]') {
                  try {
                    const d = JSON.parse(line.slice(6));
                    fullText += d.choices?.[0]?.text || d.choices?.[0]?.delta?.content || '';
                    setActiveStreams(prev => ({ ...prev, [modelId]: fullText }));
                  } catch { /* ignore parse errors */ }
                }
              }
            }
          }
        } else {
          const data = await res.json();
          fullText = data?.choices?.[0]?.message?.content || data?.choices?.[0]?.text || '(no output)';
        }

        const result: CompletionResult = {
          model: modelId, text: fullText || '(no output)',
          tokens: fullText.split(' ').length,
          duration: Date.now() - startTime,
          timestamp: Date.now(),
        };
        setResults(prev => [result, ...prev].slice(0, 50));
      } catch (e: any) {
        setResults(prev => [{
          model: modelId, text: `Error: ${e.message}`,
          tokens: 0, duration: Date.now() - startTime, timestamp: Date.now(),
        }, ...prev].slice(0, 50));
      }
    }
    setActiveStreams({});
    setRunning(false);
  }, [prompt, selectedModels, systemPrompt, temperature, maxTokens, topP, stream, setResults]);

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
          <Bot size={20} className="text-[#4fc3f7]" />
          Playground
        </h2>
        <div className="flex gap-2">
          <button onClick={() => setResults([])} className="aurelius-btn-outline flex items-center gap-1.5 text-xs">
            <Trash2 size={12} /> Clear
          </button>
          <button onClick={() => setShowSettings(!showSettings)} className="aurelius-btn-outline flex items-center gap-1.5 text-xs">
            <Settings size={12} /> {showSettings ? 'Hide' : 'Show'} Settings
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-4">
        <div className="lg:col-span-2 space-y-3">
          <Textarea
            value={prompt}
            onChange={e => setPrompt(e.target.value)}
            placeholder="Enter your prompt here..."
            rows={6}
          />
          <div className="flex gap-2">
            <button
              onClick={handleSubmit}
              disabled={running || !prompt.trim() || selectedModels.length === 0}
              className="aurelius-btn-primary flex items-center gap-2 text-sm"
            >
              {running ? <Square size={14} /> : <Play size={14} />}
              {running ? 'Running...' : `Run (${selectedModels.length} model${selectedModels.length > 1 ? 's' : ''})`}
            </button>
          </div>
        </div>

        {showSettings && (
          <div className="space-y-4">
            <div className="aurelius-card p-4">
              <h3 className="text-xs font-bold text-[#e0e0e0] uppercase tracking-wider mb-3 flex items-center gap-1.5">
                <Cpu size={12} className="text-[#4fc3f7]" /> Models
              </h3>
              <div className="space-y-1.5">
                {AVAILABLE_MODELS.map(m => (
                  <label key={m.id} className="flex items-center gap-2 cursor-pointer">
                    <input type="checkbox" checked={selectedModels.includes(m.id)}
                      onChange={() => toggleModel(m.id)}
                      className="w-3.5 h-3.5 rounded border-[#2d2d44] bg-[#0f0f1a] text-[#4fc3f7]" />
                    <span className="text-xs text-[#e0e0e0]">{m.name}</span>
                  </label>
                ))}
              </div>
            </div>

            <div className="aurelius-card p-4 space-y-4">
              <h3 className="text-xs font-bold text-[#e0e0e0] uppercase tracking-wider flex items-center gap-1.5">
                <Sliders size={12} className="text-[#4fc3f7]" /> Parameters
              </h3>
              <div>
                <label className="flex justify-between text-xs text-[#9e9eb0] mb-1">
                  <span><Thermometer size={10} className="inline mr-1" />Temperature</span>
                  <span>{temperature}</span>
                </label>
                <input type="range" min={0} max={2} step={0.05} value={temperature}
                  onChange={e => setTemperature(parseFloat(e.target.value))}
                  className="w-full accent-[#4fc3f7]" />
              </div>
              <div>
                <label className="flex justify-between text-xs text-[#9e9eb0] mb-1">
                  <span><Hash size={10} className="inline mr-1" />Max Tokens</span>
                  <span>{maxTokens}</span>
                </label>
                <input type="range" min={64} max={4096} step={64} value={maxTokens}
                  onChange={e => setMaxTokens(parseInt(e.target.value))}
                  className="w-full accent-[#4fc3f7]" />
              </div>
              <div>
                <label className="flex justify-between text-xs text-[#9e9eb0] mb-1">
                  <span>Top P</span>
                  <span>{topP}</span>
                </label>
                <input type="range" min={0} max={1} step={0.05} value={topP}
                  onChange={e => setTopP(parseFloat(e.target.value))}
                  className="w-full accent-[#4fc3f7]" />
              </div>
              <div className="flex items-center justify-between">
                <span className="text-xs text-[#9e9eb0]">Stream output</span>
                <Toggle checked={stream} onChange={setStream} />
              </div>
              <div>
                <label className="text-xs text-[#9e9eb0] mb-1 block">System Prompt</label>
                <textarea value={systemPrompt} onChange={e => setSystemPrompt(e.target.value)}
                  className="w-full bg-[#0f0f1a] border border-[#2d2d44] rounded text-xs text-[#e0e0e0] p-2 h-20 resize-none" />
              </div>
            </div>
          </div>
        )}
      </div>

      {results.length > 0 && (
        <div className="space-y-3">
          <h3 className="text-sm font-semibold text-[#e0e0e0]">Output</h3>
          {results.slice(0, 10).map((result, i) => (
            <motion.div key={i} initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
              className="aurelius-card p-4">
              <div className="flex items-center justify-between mb-2">
                <div className="flex items-center gap-2">
                  <Bot size={14} className="text-[#4fc3f7]" />
                  <span className="text-sm font-medium text-[#e0e0e0]">{
                    AVAILABLE_MODELS.find(m => m.id === result.model)?.name || result.model
                  }</span>
                  <span className="text-[10px] text-[#9e9eb0]">{(result.duration / 1000).toFixed(1)}s · {result.tokens} tokens</span>
                </div>
                <button onClick={() => { navigator.clipboard.writeText(result.text); toast('Copied', 'success'); }}
                  className="text-[#9e9eb0] hover:text-[#e0e0e0]"><Copy size={12} /></button>
              </div>
              <pre className="text-xs text-[#9e9eb0] whitespace-pre-wrap font-sans leading-relaxed max-h-60 overflow-y-auto">
                {activeStreams[result.model] || result.text}
              </pre>
            </motion.div>
          ))}
          <div ref={bottomRef} />
        </div>
      )}
    </div>
  );
}
