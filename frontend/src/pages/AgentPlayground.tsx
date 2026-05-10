import { useState, useRef, useEffect, useCallback } from 'react';
import { motion } from 'framer-motion';
import {
  Play, Square, RotateCcw, Terminal, MessageSquare,
  Bot, Cpu, Clock, Loader2, Save, Trash2, Code,
  Zap, AlertCircle, CheckCircle2, XCircle, Activity,
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

// Provider stats mirror ProviderChain.ProviderStats
interface ProviderStat {
  name: string;
  calls: number;
  successes: number;
  failures: number;
  consecutive_failures: number;
  success_rate: number;
  circuit_open: boolean;
  last_error: string;
}

// Available providers in the chain
const PROVIDER_NAMES = ['mock', 'vllm', 'agentic'] as const;
type ProviderName = typeof PROVIDER_NAMES[number];

// Mock generate_fn factory - simulates provider behavior
const createMockGenerateFn = (name: ProviderName, failRate: number = 0.1) => {
  return (_messages: { role: string; content: string }[]): string => {
    // Simulate occasional failures for circuit breaker demo
    if (Math.random() < failRate) {
      throw new Error(`[${name}] Simulated provider failure`);
    }
    // Return a mock response
    return `[${name.toUpperCase()}] Processed request successfully`;
  };
};

export default function AgentPlayground() {
  const [prompt, setPrompt] = useState('');
  const [agentType, setAgentType] = useState('coding');
  const [temperature, setTemperature] = useState(0.7);
  const [maxTokens, setMaxTokens] = useState(512);
  const [results, setResults] = useState<TestResult[]>([]);
  const [running, setRunning] = useState(false);
  const [streamingText, setStreamingText] = useState('');
  const [savedPrompts, setSavedPrompts] = useState<string[]>([]);

  // Provider Chain mode state
  const [playgroundMode, setPlaygroundMode] = useState<'single' | 'chain'>('single');
  const [providerStats, setProviderStats] = useState<Record<string, ProviderStat>>({
    mock: { name: 'mock', calls: 0, successes: 0, failures: 0, consecutive_failures: 0, success_rate: 0, circuit_open: false, last_error: '' },
    vllm: { name: 'vllm', calls: 0, successes: 0, failures: 0, consecutive_failures: 0, success_rate: 0, circuit_open: false, last_error: '' },
    agentic: { name: 'agentic', calls: 0, successes: 0, failures: 0, consecutive_failures: 0, success_rate: 0, circuit_open: false, last_error: '' },
  });
  const [classificationResult, setClassificationResult] = useState<{ category: string; confidence: string } | null>(null);
  const bottomRef = useRef<HTMLDivElement>(null);

  // Failure thresholds per provider (for circuit breaker simulation)
  const [failureThresholds] = useState<Record<string, number>>({
    mock: 3,
    vllm: 3,
    agentic: 3,
  });

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [results, streamingText]);

  // Execute provider chain with fallback logic + circuit breaker simulation
  const executeProviderChain = useCallback(async (messages: { role: string; content: string }[]) => {
    const triedProviders: string[] = [];
    let lastError = '';

    // Try providers in order (mock -> vllm -> agentic)
    for (const providerName of PROVIDER_NAMES) {
      const stats = providerStats[providerName];

      // Skip if circuit is open
      if (stats.circuit_open) {
        setResults(prev => [...prev, {
          type: 'thought',
          content: `Circuit breaker OPEN for ${providerName}, skipping...`,
          timestamp: Date.now()
        }]);
        continue;
      }

      triedProviders.push(providerName);
      const generateFn = createMockGenerateFn(providerName, 0.15); // 15% fail rate

      setResults(prev => [...prev, {
        type: 'thought',
        content: `Trying provider: ${providerName}...`,
        timestamp: Date.now()
      }]);

      try {
        const output = generateFn(messages);

        // Update stats on success
        setProviderStats(prev => {
          const s = prev[providerName];
          const newCalls = s.calls + 1;
          const newSuccesses = s.successes + 1;
          return {
            ...prev,
            [providerName]: {
              ...s,
              calls: newCalls,
              successes: newSuccesses,
              consecutive_failures: 0,
              circuit_open: false,
              success_rate: newSuccesses / newCalls,
            },
          };
        });

        setResults(prev => [...prev, {
          type: 'output',
          content: `Provider ${providerName} succeeded: ${output}`,
          timestamp: Date.now()
        }]);

        return { providerName, output, triedProviders };
      } catch (e: any) {
        lastError = `${providerName}: ${e.message}`;

        // Update stats on failure AND check circuit breaker in same callback
        setProviderStats(prev => {
          const s = prev[providerName];
          const newCalls = s.calls + 1;
          const newFailures = s.failures + 1;
          const newConsecutive = s.consecutive_failures + 1;
          const shouldOpen = newConsecutive >= failureThresholds[providerName];

          if (shouldOpen) {
            setResults(r => [...r, {
              type: 'thought',
              content: `Circuit breaker TRIPPED for ${providerName}!`,
              timestamp: Date.now(),
            }]);
          }

          return {
            ...prev,
            [providerName]: {
              ...s,
              calls: newCalls,
              failures: newFailures,
              consecutive_failures: newConsecutive,
              last_error: lastError,
              circuit_open: shouldOpen,
              success_rate: s.successes / newCalls,
            },
          };
        });

        setResults(prev => [...prev, {
          type: 'error',
          content: `${providerName} failed: ${e.message}`,
          timestamp: Date.now(),
        }]);
      }
    }

    // All providers failed
    setResults(prev => [...prev, {
      type: 'error',
      content: `All providers failed. Tried: ${triedProviders.join(', ')}. Last error: ${lastError}`,
      timestamp: Date.now()
    }]);

    return null;
  }, [providerStats, failureThresholds]);

  // Classify task using AuxiliaryClient pattern
  const classifyTask = useCallback((text: string) => {
    // Simple keyword-based classification for demo
    const categories = ['coding', 'research', 'general'];
    const lower = text.toLowerCase();

    if (lower.includes('code') || lower.includes('function') || lower.includes('debug')) {
      return { category: 'coding', confidence: 'high' };
    } else if (lower.includes('research') || lower.includes('find') || lower.includes('search')) {
      return { category: 'research', confidence: 'medium' };
    }
    return { category: 'general', confidence: 'low' };
  }, []);

  const handleRun = async () => {
    if (!prompt.trim()) return;
    setRunning(true);
    setStreamingText('');
    setClassificationResult(null);

    const startTime = Date.now();
    const messages = [{ role: 'user', content: prompt }];

    if (playgroundMode === 'chain') {
      // Provider Chain mode
      setResults(prev => [...prev, { type: 'thought', content: 'Running in Provider Chain mode...', timestamp: Date.now() }]);

      // Show task classification via AuxiliaryClient pattern
      const classification = classifyTask(prompt);
      setClassificationResult(classification);
      setResults(prev => [...prev, {
        type: 'tool_result',
        content: `AuxiliaryClient.classify() → Category: ${classification.category} (confidence: ${classification.confidence})`,
        timestamp: Date.now()
      }]);

      // Execute the provider chain
      await executeProviderChain(messages);

      // Show final stats
      setResults(prev => [...prev, {
        type: 'thought',
        content: 'ProviderChain.stats() snapshot:',
        timestamp: Date.now()
      }]);

      Object.values(providerStats).forEach(s => {
        setResults(prev => [...prev, {
          type: 'tool_result',
          content: `  ${s.name}: calls=${s.calls}, successes=${s.successes}, failures=${s.failures}, success_rate=${(s.success_rate * 100).toFixed(1)}%, circuit_open=${s.circuit_open}`,
          timestamp: Date.now(),
        }]);
      });
    } else {
      // Single Provider mode (original behavior)
      setResults(prev => [...prev, { type: 'thought', content: `Routing to ${agentType} agent...`, timestamp: Date.now() }]);

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
    }

    setStreamingText('');
    setRunning(false);
  };

  const resetCircuit = (providerName: string) => {
    setProviderStats(prev => ({
      ...prev,
      [providerName]: {
        ...prev[providerName],
        consecutive_failures: 0,
        circuit_open: false,
        last_error: '',
      },
    }));
    setResults(prev => [...prev, {
      type: 'thought',
      content: `Circuit breaker reset for ${providerName}`,
      timestamp: Date.now()
    }]);
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
                  r.type === 'tool_result' ? 'border-violet-500/20 bg-violet-500/5' :
                  'border-[#2d2d44] bg-[#0f0f1a]'
                }`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span className={`text-[10px] font-bold uppercase ${
                    r.type === 'error' ? 'text-rose-400' :
                    r.type === 'tool_call' ? 'text-amber-400' :
                    r.type === 'output' ? 'text-emerald-400' :
                    r.type === 'tool_result' ? 'text-violet-400' : 'text-[#4fc3f7]'
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
              <span className="text-[10px] text-[#9e9eb0]">
                {playgroundMode === 'chain' ? 'Provider Chain' : agentType} · T={temperature} · max={maxTokens}
              </span>
            </div>
          </div>
        </div>
      </div>

      <div className="w-72 hidden lg:flex flex-col space-y-4">
        {/* Mode Selector */}
        <div className="aurelius-card p-4 space-y-3">
          <h3 className="text-xs font-bold text-[#e0e0e0] uppercase tracking-wider flex items-center gap-2">
            <Activity size={12} className="text-[#4fc3f7]" /> Mode
          </h3>
          <div className="flex gap-2">
            <button
              onClick={() => setPlaygroundMode('single')}
              className={`flex-1 py-1.5 text-xs rounded-lg transition-colors ${
                playgroundMode === 'single'
                  ? 'bg-[#4fc3f7]/20 text-[#4fc3f7] border border-[#4fc3f7]/30'
                  : 'bg-[#0f0f1a] text-[#9e9eb0] border border-[#2d2d44] hover:text-[#e0e0e0]'
              }`}
            >
              Single Provider
            </button>
            <button
              onClick={() => setPlaygroundMode('chain')}
              className={`flex-1 py-1.5 text-xs rounded-lg transition-colors ${
                playgroundMode === 'chain'
                  ? 'bg-[#4fc3f7]/20 text-[#4fc3f7] border border-[#4fc3f7]/30'
                  : 'bg-[#0f0f1a] text-[#9e9eb0] border border-[#2d2d44] hover:text-[#e0e0e0]'
              }`}
            >
              Provider Chain
            </button>
          </div>
        </div>

        {/* Provider Chain Panel */}
        {playgroundMode === 'chain' && (
          <>
            {/* Provider Stats Panel */}
            <div className="aurelius-card p-4 space-y-3">
              <h3 className="text-xs font-bold text-[#e0e0e0] uppercase tracking-wider flex items-center gap-2">
                <Zap size={12} className="text-amber-400" /> Provider Chain
              </h3>

              <div className="space-y-2">
                {PROVIDER_NAMES.map(name => {
                  const stat = providerStats[name];
                  return (
                    <div key={name} className="bg-[#0f0f1a] rounded-lg p-2 border border-[#2d2d44]">
                      <div className="flex items-center justify-between mb-1">
                        <div className="flex items-center gap-1.5">
                          <span className="text-xs font-medium text-[#e0e0e0] uppercase">{name}</span>
                          {stat.circuit_open && (
                            <span className="flex items-center gap-0.5 text-[10px] text-rose-400">
                              <AlertCircle size={10} /> CB OPEN
                            </span>
                          )}
                        </div>
                        <button
                          onClick={() => resetCircuit(name)}
                          className="text-[10px] text-[#9e9eb0] hover:text-[#4fc3f7] transition-colors flex items-center gap-0.5"
                          title="Reset circuit breaker"
                        >
                          <RotateCcw size={10} /> Reset
                        </button>
                      </div>

                      <div className="grid grid-cols-4 gap-1 text-[10px]">
                        <div className="text-center">
                          <div className="text-[#9e9eb0]">Calls</div>
                          <div className="text-[#e0e0e0] font-medium">{stat.calls}</div>
                        </div>
                        <div className="text-center">
                          <div className="text-[#9e9eb0]">OK</div>
                          <div className="text-emerald-400 font-medium">{stat.successes}</div>
                        </div>
                        <div className="text-center">
                          <div className="text-[#9e9eb0]">Fail</div>
                          <div className="text-rose-400 font-medium">{stat.failures}</div>
                        </div>
                        <div className="text-center">
                          <div className="text-[#9e9eb0]">Rate</div>
                          <div className={`font-medium ${stat.success_rate >= 0.8 ? 'text-emerald-400' : stat.success_rate >= 0.5 ? 'text-amber-400' : 'text-rose-400'}`}>
                            {(stat.success_rate * 100).toFixed(0)}%
                          </div>
                        </div>
                      </div>

                      {stat.last_error && (
                        <div className="mt-1 text-[10px] text-rose-400/70 truncate">
                          Last error: {stat.last_error}
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            </div>

            {/* AuxiliaryClient Panel */}
            <div className="aurelius-card p-4 space-y-3">
              <h3 className="text-xs font-bold text-[#e0e0e0] uppercase tracking-wider flex items-center gap-2">
                <Cpu size={12} className="text-violet-400" /> AuxiliaryClient
              </h3>

              <div className="text-[10px] text-[#9e9eb0] space-y-1">
                <div className="flex items-center gap-1 text-[#9e9eb0]">
                  <span className="text-violet-400">classify()</span>
                  <span>Task classification for routing</span>
                </div>
                {classificationResult ? (
                  <div className="bg-[#0f0f1a] rounded-lg p-2 border border-violet-500/20">
                    <div className="text-[10px] text-violet-400 mb-1">Result:</div>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-[#e0e0e0]">Category: <span className="text-violet-400 font-medium">{classificationResult.category}</span></span>
                      <span className="text-xs text-[#9e9eb0]">Confidence: <span className={`font-medium ${classificationResult.confidence === 'high' ? 'text-emerald-400' : classificationResult.confidence === 'medium' ? 'text-amber-400' : 'text-[#9e9eb0]'}`}>{classificationResult.confidence}</span></span>
                    </div>
                  </div>
                ) : (
                  <div className="text-[10px] text-[#9e9eb0]/50 italic">
                    Run a prompt to see classification results
                  </div>
                )}
              </div>
            </div>
          </>
        )}

        {/* Single Provider Config */}
        {playgroundMode === 'single' && (
          <>
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
          </>
        )}
      </div>
    </div>
  );
}
