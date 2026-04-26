import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import {
  Bot,
  ArrowLeft,
  Activity,
  Clock,
  Zap,
  Pause,
  Play,
  Loader2,
  AlertTriangle,
  RefreshCw,
} from 'lucide-react';
import { useApi } from '../hooks/useApi';
import { useToast } from '../components/ToastProvider';

interface AgentDetailData {
  id: string;
  state: string;
  metrics?: Record<string, number | string>;
}

const agentRoles: Record<string, string> = {
  hermes: 'Notification Router',
  openclaw: 'Task Orchestrator',
  cerebrum: 'Memory Manager',
  vigil: 'Security Warden',
  default: 'System Agent',
};

function timeAgo(ts: number): string {
  const diff = Date.now() / 1000 - ts;
  if (diff < 60) return 'Just now';
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} hr ago`;
  return `${Math.floor(diff / 86400)} days ago`;
}

export default function AgentDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const { toast } = useToast();
  const [history, setHistory] = useState<{ timestamp: number; state: string }[]>([]);

  const {
    data,
    loading,
    error,
    refresh,
  } = useApi<AgentDetailData>(`/agents/${id}`, {
    refreshInterval: 5000,
  });

  useEffect(() => {
    if (error) {
      toast(`Failed to load agent: ${error.message}`, 'error');
    }
  }, [error, toast]);

  useEffect(() => {
    if (data) {
      setHistory((prev) => {
        const next = [...prev, { timestamp: Date.now() / 1000, state: data.state }];
        if (next.length > 20) return next.slice(-20);
        return next;
      });
    }
  }, [data?.state]);

  if (!id) return null;

  const stateUpper = data?.state.toUpperCase() || 'UNKNOWN';
  const isActive = stateUpper === 'ACTIVE' || stateUpper === 'RUNNING';
  const isIdle = stateUpper === 'IDLE';

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <button
          onClick={() => navigate('/')}
          className="p-2 rounded-lg text-aurelius-muted hover:text-aurelius-text hover:bg-aurelius-border/40 transition-colors"
        >
          <ArrowLeft size={18} />
        </button>
        <h2 className="text-lg font-bold text-aurelius-text flex items-center gap-2">
          <Bot size={20} className="text-aurelius-accent" />
          {id.replace(/^\w/, (c) => c.toUpperCase())}
        </h2>
        <button
          onClick={refresh}
          disabled={loading}
          className="ml-auto aurelius-btn-outline flex items-center gap-2 text-sm disabled:opacity-50"
        >
          {loading ? <Loader2 size={14} className="animate-spin" /> : <RefreshCw size={14} />}
          Refresh
        </button>
      </div>

      {error && (
        <div className="aurelius-card border-rose-500/30 bg-rose-500/5 text-rose-300">
          <AlertTriangle size={18} className="inline mr-2" />
          {error.message}
        </div>
      )}

      {loading && !data && (
        <div className="aurelius-card text-center py-12 text-aurelius-muted">
          <Loader2 size={32} className="mx-auto mb-3 animate-spin opacity-60" />
          <p>Loading agent details...</p>
        </div>
      )}

      {data && (
        <>
          {/* Status Card */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <div className="aurelius-card space-y-2">
              <div className="flex items-center gap-2">
                {isActive ? <Play size={16} className="text-emerald-400" /> : isIdle ? <Pause size={16} className="text-amber-400" /> : <Zap size={16} className="text-rose-400" />}
                <span className="text-xs text-aurelius-muted uppercase tracking-wider">State</span>
              </div>
              <p className={`text-2xl font-bold ${isActive ? 'text-emerald-400' : isIdle ? 'text-amber-400' : 'text-rose-400'}`}>
                {data.state}
              </p>
            </div>
            <div className="aurelius-card space-y-2">
              <div className="flex items-center gap-2">
                <Activity size={16} className="text-aurelius-accent" />
                <span className="text-xs text-aurelius-muted uppercase tracking-wider">Role</span>
              </div>
              <p className="text-2xl font-bold text-aurelius-text">
                {agentRoles[id.toLowerCase()] || agentRoles.default}
              </p>
            </div>
            <div className="aurelius-card space-y-2">
              <div className="flex items-center gap-2">
                <Clock size={16} className="text-aurelius-accent" />
                <span className="text-xs text-aurelius-muted uppercase tracking-wider">Observations</span>
              </div>
              <p className="text-2xl font-bold text-aurelius-text">{history.length}</p>
            </div>
          </div>

          {/* Metrics */}
          {data.metrics && Object.keys(data.metrics).length > 0 && (
            <div className="aurelius-card space-y-4">
              <h3 className="text-sm font-semibold text-aurelius-text uppercase tracking-wider">Metrics</h3>
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {Object.entries(data.metrics).map(([key, value]) => (
                  <div key={key} className="bg-aurelius-bg border border-aurelius-border rounded-lg p-3">
                    <p className="text-xs text-aurelius-muted uppercase tracking-wider">{key}</p>
                    <p className="text-lg font-bold text-aurelius-text mt-0.5">{String(value)}</p>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* State History */}
          <div className="aurelius-card space-y-4">
            <h3 className="text-sm font-semibold text-aurelius-text uppercase tracking-wider flex items-center gap-2">
              <Activity size={16} className="text-aurelius-accent" />
              State History
            </h3>
            {history.length === 0 ? (
              <p className="text-sm text-aurelius-muted text-center py-6">No state changes observed yet.</p>
            ) : (
              <div className="space-y-2 max-h-[300px] overflow-y-auto">
                {history.slice().reverse().map((h, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between px-3 py-2 rounded-lg bg-aurelius-bg border border-aurelius-border"
                  >
                    <div className="flex items-center gap-2">
                      <div
                        className={`w-2 h-2 rounded-full ${
                          h.state.toUpperCase() === 'ACTIVE' || h.state.toUpperCase() === 'RUNNING'
                            ? 'bg-emerald-400'
                            : h.state.toUpperCase() === 'IDLE'
                            ? 'bg-amber-400'
                            : 'bg-rose-400'
                        }`}
                      />
                      <span className="text-sm text-aurelius-text">{h.state}</span>
                    </div>
                    <span className="text-xs text-aurelius-muted">{timeAgo(h.timestamp)}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
