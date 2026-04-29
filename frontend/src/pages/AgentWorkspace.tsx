import { useState, useEffect, useRef } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Cpu, ArrowLeft, Terminal, MessageSquare, Square,
  Loader2, AlertCircle, CheckCircle, Clock, Activity,
} from 'lucide-react';

interface Agent {
  id: string; name: string; role: string; capabilities: string[];
  state: string; created: number; lastHeartbeat: number;
  metrics: { tasksCompleted: number; avgLatencyMs: number; errorRate: number };
}

interface StreamEvent {
  type: string;
  data: unknown;
  timestamp: number;
}

const eventIcons: Record<string, typeof Terminal> = {
  thought: MessageSquare,
  tool_call: Terminal,
  tool_result: CheckCircle,
  action: Activity,
  error: AlertCircle,
};

export default function AgentWorkspace() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [agent, setAgent] = useState<Agent | null>(null);
  const [events, setEvents] = useState<StreamEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [loading, setLoading] = useState(true);
  const bottomRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    fetch(`/api/agents/${id}`)
      .then(r => r.json())
      .then(d => { setAgent(d.agent); setLoading(false); })
      .catch(() => setLoading(false));
  }, [id]);

  useEffect(() => {
    if (!id) return;
    const es = new EventSource(`/api/agents/${id}/stream`);
    es.onopen = () => setConnected(true);

    const handlers: Record<string, (d: any) => void> = {
      connected: () => setConnected(true),
      heartbeat: (data: Agent) => setAgent(data),
      created: (data: Agent) => setAgent(data),
      terminated: () => setAgent(prev => prev ? { ...prev, state: 'terminated' } : null),
    };

    for (const [event, handler] of Object.entries(handlers)) {
      es.addEventListener(event, (e: MessageEvent) => handler(JSON.parse(e.data)));
    }

    es.addEventListener('message', (e: MessageEvent) => {
      try {
        const data = JSON.parse(e.data);
        setEvents(prev => [...prev.slice(-200), { type: 'action', data, timestamp: Date.now() }]);
      } catch { /* ignore non-json */ }
    });

    es.onerror = () => setConnected(false);
    return () => es.close();
  }, [id]);

  useEffect(() => { bottomRef.current?.scrollIntoView({ behavior: 'smooth' }); }, [events]);

  if (loading) return <div className="flex justify-center py-20"><Loader2 size={32} className="animate-spin text-[#4fc3f7]" /></div>;
  if (!agent) return <div className="text-center py-20 text-[#9e9eb0]">Agent not found</div>;

  const stateStyle: Record<string, string> = {
    idle: 'text-gray-400', active: 'text-emerald-400', busy: 'text-amber-400',
    error: 'text-rose-400', terminated: 'text-gray-600',
  };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-3">
          <button onClick={() => navigate('/agents')} className="text-[#9e9eb0] hover:text-[#e0e0e0]">
            <ArrowLeft size={20} />
          </button>
          <div className="flex items-center gap-3">
            <div className={`w-10 h-10 rounded-lg flex items-center justify-center bg-[#4fc3f7]/10 text-[#4fc3f7]`}>
              <Cpu size={20} />
            </div>
            <div>
              <h2 className="text-lg font-bold text-[#e0e0e0]">{agent.name}</h2>
              <p className="text-xs text-[#9e9eb0]">{agent.role} · {agent.capabilities.join(', ')}</p>
            </div>
          </div>
        </div>
        <div className="flex items-center gap-3">
          <span className={`flex items-center gap-1.5 text-xs ${connected ? 'text-emerald-400' : 'text-rose-400'}`}>
            <span className={`w-2 h-2 rounded-full ${connected ? 'bg-emerald-400 animate-pulse' : 'bg-rose-400'}`} />
            {connected ? 'Live' : 'Disconnected'}
          </span>
          <span className={`text-xs font-medium px-2 py-1 rounded-full border ${stateStyle[agent.state]} bg-[#0f0f1a] border-[#2d2d44]`}>
            {agent.state}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-4 gap-3">
        {[
          { label: 'Tasks', value: agent.metrics.tasksCompleted, icon: CheckCircle, color: 'text-[#4fc3f7]' },
          { label: 'Latency', value: `${agent.metrics.avgLatencyMs.toFixed(0)}ms`, icon: Clock, color: 'text-amber-400' },
          { label: 'Error Rate', value: `${(agent.metrics.errorRate * 100).toFixed(0)}%`, icon: AlertCircle, color: 'text-rose-400' },
          { label: 'Uptime', value: `${Math.floor((Date.now() - agent.created) / 1000)}s`, icon: Activity, color: 'text-emerald-400' },
        ].map(s => (
          <div key={s.label} className="aurelius-card text-center py-3">
            <s.icon size={16} className={`mx-auto mb-1 ${s.color}`} />
            <p className={`text-lg font-bold ${s.color}`}>{s.value}</p>
            <p className="text-[10px] text-[#9e9eb0] uppercase tracking-wider">{s.label}</p>
          </div>
        ))}
      </div>

      <div className="aurelius-card p-4">
        <div className="flex items-center justify-between mb-3">
          <h3 className="text-sm font-semibold text-[#e0e0e0] flex items-center gap-2">
            <Terminal size={14} className="text-[#4fc3f7]" />
            Execution Stream
          </h3>
          <span className="text-[10px] text-[#9e9eb0]">{events.length} events</span>
        </div>
        <div className="h-80 overflow-y-auto space-y-1 font-mono text-xs bg-[#0a0a14] rounded-lg p-3 border border-[#2d2d44]">
          {events.length === 0 && (
            <p className="text-[#9e9eb0] text-center py-8">Waiting for events...</p>
          )}
          {events.map((ev, i) => {
            const Icon = eventIcons[ev.type] || Terminal;
            return (
              <div key={i} className="flex items-start gap-2 text-[#9e9eb0] hover:text-[#e0e0e0]">
                <Icon size={12} className="mt-0.5 shrink-0 text-[#4fc3f7]" />
                <span className="text-[#4fc3f7]/60">{new Date(ev.timestamp).toISOString().slice(11, 19)}</span>
                <span className="truncate">{JSON.stringify(ev.data).slice(0, 200)}</span>
              </div>
            );
          })}
          <div ref={bottomRef} />
        </div>
      </div>
    </div>
  );
}
