import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, Terminal, MessageSquare, CheckCircle, AlertCircle, Activity, Clock, Loader2 } from 'lucide-react';

interface TraceStep {
  id: string; type: string; timestamp: number;
  content: string; metadata?: Record<string, unknown>; duration?: number;
}

interface Trace {
  id: string; agentId: string; agentName: string; task: string;
  status: string; steps: TraceStep[]; startedAt: number;
  completedAt?: number; totalDuration?: number; stepCount: number; tokenCount?: number;
}

const stepIcons: Record<string, typeof Terminal> = {
  thought: MessageSquare, tool_call: Terminal, tool_result: CheckCircle,
  action: Activity, error: AlertCircle, observation: Clock,
};

const stepColors: Record<string, string> = {
  thought: 'text-[#4fc3f7]', tool_call: 'text-amber-400',
  tool_result: 'text-emerald-400', action: 'text-violet-400',
  error: 'text-rose-400', observation: 'text-gray-400',
};

export default function TraceDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [trace, setTrace] = useState<Trace | null>(null);
  const [loading, setLoading] = useState(true);
  const [expandedStep, setExpandedStep] = useState<string | null>(null);

  useEffect(() => {
    fetch(`/api/traces/${id}`)
      .then(r => r.json())
      .then(d => { setTrace(d.trace); setLoading(false); })
      .catch(() => setLoading(false));
  }, [id]);

  if (loading) return <div className="flex justify-center py-20"><Loader2 size={32} className="animate-spin text-[#4fc3f7]" /></div>;
  if (!trace) return <div className="text-center py-20 text-[#9e9eb0]">Trace not found</div>;

  const statusColor: Record<string, string> = {
    completed: 'text-emerald-400', failed: 'text-rose-400',
    running: 'text-[#4fc3f7]', truncated: 'text-amber-400',
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="flex items-center gap-3">
        <button onClick={() => navigate('/traces')} className="text-[#9e9eb0] hover:text-[#e0e0e0]"><ArrowLeft size={20} /></button>
        <div className="flex-1 min-w-0">
          <h2 className="text-lg font-bold text-[#e0e0e0] truncate">{trace.task}</h2>
          <p className="text-xs text-[#9e9eb0]">{trace.agentName} · {trace.stepCount} steps{trace.totalDuration ? ` · ${(trace.totalDuration / 1000).toFixed(1)}s` : ''}</p>
        </div>
        <span className={`text-xs font-bold px-3 py-1 rounded-full border ${statusColor[trace.status]} bg-[#0f0f1a] border-[#2d2d44]`}>{trace.status}</span>
      </div>

      <div className="grid grid-cols-4 gap-3 text-center">
        <div><p className="text-lg font-bold text-[#e0e0e0]">{trace.stepCount}</p><p className="text-[10px] text-[#9e9eb0] uppercase tracking-wider">Steps</p></div>
        <div><p className="text-lg font-bold text-[#e0e0e0]">{trace.totalDuration ? `${(trace.totalDuration / 1000).toFixed(1)}s` : '-'}</p><p className="text-[10px] text-[#9e9eb0] uppercase tracking-wider">Duration</p></div>
        <div><p className="text-lg font-bold text-[#e0e0e0]">{trace.tokenCount || '-'}</p><p className="text-[10px] text-[#9e9eb0] uppercase tracking-wider">Tokens</p></div>
        <div><p className="text-lg font-bold text-[#e0e0e0]">{new Date(trace.startedAt).toLocaleTimeString()}</p><p className="text-[10px] text-[#9e9eb0] uppercase tracking-wider">Started</p></div>
      </div>

      <div className="relative">
        <div className="absolute left-4 top-0 bottom-0 w-px bg-[#2d2d44]" />
        <div className="space-y-0">
          {trace.steps.map((step) => {
            const Icon = stepIcons[step.type] || Terminal;
            const color = stepColors[step.type] || 'text-gray-400';
            const isExpanded = expandedStep === step.id;
            const time = new Date(step.timestamp).toLocaleTimeString();

            return (
              <div key={step.id} className="relative pl-10 pb-4">
                <div className={`absolute left-2.5 w-3 h-3 rounded-full border-2 border-[#2d2d44] ${color} bg-[#0a0a14] flex items-center justify-center`} style={{ top: '4px' }}>
                  <Icon size={8} />
                </div>
                <div
                  onClick={() => setExpandedStep(isExpanded ? null : step.id)}
                  className="aurelius-card p-3 hover:border-[#4fc3f7]/20 cursor-pointer transition-all"
                >
                  <div className="flex items-center justify-between mb-1">
                    <div className="flex items-center gap-2">
                      <Icon size={12} className={color} />
                      <span className={`text-[10px] font-bold uppercase tracking-wider ${color}`}>{step.type}</span>
                      {step.duration && <span className="text-[10px] text-[#9e9eb0]">({step.duration}ms)</span>}
                    </div>
                    <span className="text-[10px] text-[#9e9eb0]">{time}</span>
                  </div>
                  <p className={`text-xs ${isExpanded ? '' : 'line-clamp-2'} text-[#9e9eb0] font-mono`}>
                    {step.content}
                  </p>
                  {isExpanded && step.metadata && Object.keys(step.metadata).length > 0 && (
                    <pre className="mt-2 text-[10px] text-[#4fc3f7]/80 bg-[#0a0a14] p-2 rounded overflow-x-auto">
                      {JSON.stringify(step.metadata, null, 2)}
                    </pre>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
}
