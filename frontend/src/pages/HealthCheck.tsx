import { useState, useEffect } from 'react';
import {
  HeartPulse, CheckCircle2, AlertTriangle, XCircle, Loader2, RefreshCw,
  Server, Database, Bot, Wifi, Shield,
} from 'lucide-react';
import { useToast } from '../components/ToastProvider';
import { api } from '../api/AureliusClient';

interface HealthCheck {
  name: string;
  status: 'healthy' | 'warning' | 'critical';
  message: string;
  latency_ms: number;
  lastChecked: string;
}

const statusConfig = {
  healthy: { icon: CheckCircle2, color: 'text-emerald-400', bg: 'bg-emerald-500/10', border: 'border-emerald-500/20', label: 'Healthy' },
  warning: { icon: AlertTriangle, color: 'text-amber-400', bg: 'bg-amber-500/10', border: 'border-amber-500/20', label: 'Warning' },
  critical: { icon: XCircle, color: 'text-rose-400', bg: 'bg-rose-500/10', border: 'border-rose-500/20', label: 'Critical' },
};

const checkIcons: Record<string, typeof Server> = { server: Server, database: Database, agents: Bot, network: Wifi, security: Shield };

function formatUptime(seconds: number): string {
  const d = Math.floor(seconds / 86400); const h = Math.floor((seconds % 86400) / 3600); const m = Math.floor((seconds % 3600) / 60);
  if (d > 0) return `${d}d ${h}h ${m}m`; if (h > 0) return `${h}h ${m}m`; return `${m}m`;
}

export default function HealthCheckPage() {
  const { toast } = useToast();
  const [manualChecks, setManualChecks] = useState<HealthCheck[]>([]);
  const [data, setData] = useState<{ overall: string; checks: HealthCheck[]; uptime_seconds: number } | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetch = async () => {
      try {
        const h = await api.getHealth();
        setData({ overall: 'healthy', checks: [], uptime_seconds: h.time });
      } catch { /* ignore */ }
      setLoading(false);
    };
    fetch();
  }, []);

  const runChecks = async () => {
    toast('Running health checks...', 'info');
    const checks: HealthCheck[] = [];

    const serverStart = performance.now();
    try { const res = await fetch('/api/health'); checks.push({ name: 'server', status: res.ok ? 'healthy' : 'critical', message: res.ok ? 'Server responding' : 'Server error', latency_ms: Math.round(performance.now() - serverStart), lastChecked: new Date().toISOString() }); }
    catch { checks.push({ name: 'server', status: 'critical', message: 'Server unreachable', latency_ms: Math.round(performance.now() - serverStart), lastChecked: new Date().toISOString() }); }

    const dbStart = performance.now();
    try { const res = await api.getMemoryLayers(); checks.push({ name: 'database', status: 'healthy', message: `${Object.keys(res.layers).length} layers accessible`, latency_ms: Math.round(performance.now() - dbStart), lastChecked: new Date().toISOString() }); }
    catch { checks.push({ name: 'database', status: 'critical', message: 'Memory unreachable', latency_ms: Math.round(performance.now() - dbStart), lastChecked: new Date().toISOString() }); }

    const agentStart = performance.now();
    try { const s = await api.getStatus(); const online = s.agents.filter((a) => a.state?.toUpperCase() === 'ACTIVE' || a.state?.toUpperCase() === 'RUNNING').length; checks.push({ name: 'agents', status: online > 0 ? 'healthy' : 'warning', message: `${online}/${s.agents.length} agents online`, latency_ms: Math.round(performance.now() - agentStart), lastChecked: new Date().toISOString() }); }
    catch { checks.push({ name: 'agents', status: 'critical', message: 'Cannot check agents', latency_ms: Math.round(performance.now() - agentStart), lastChecked: new Date().toISOString() }); }

    checks.push({ name: 'network', status: navigator.onLine ? 'healthy' : 'critical', message: navigator.onLine ? 'Connected' : 'Offline', latency_ms: 0, lastChecked: new Date().toISOString() });
    checks.push({ name: 'security', status: window.isSecureContext ? 'healthy' : 'warning', message: window.isSecureContext ? 'Secure context' : 'Insecure context', latency_ms: 0, lastChecked: new Date().toISOString() });

    setManualChecks(checks);
    const hasCritical = checks.some((c) => c.status === 'critical');
    const hasWarning = checks.some((c) => c.status === 'warning');
    if (hasCritical) toast('Critical health issues detected', 'error');
    else if (hasWarning) toast('Health warnings detected', 'warning');
    else toast('All systems healthy', 'success');
  };

  const checks = manualChecks.length > 0 ? manualChecks : (data?.checks || []);
  const overall = (data?.overall || 'healthy') as keyof typeof statusConfig;
  const overallConfig = statusConfig[overall];
  const OverallIcon = overallConfig.icon;
  const healthyCount = checks.filter((c) => c.status === 'healthy').length;
  const warningCount = checks.filter((c) => c.status === 'warning').length;
  const criticalCount = checks.filter((c) => c.status === 'critical').length;

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2"><HeartPulse size={20} className="text-[#4fc3f7]" /> Health Check</h2>
        <div className="flex items-center gap-3">
          {data && <span className="text-xs text-[#9e9eb0]">Uptime: {formatUptime(data.uptime_seconds || 0)}</span>}
          <button onClick={runChecks} className="aurelius-btn-outline flex items-center gap-2 text-sm"><RefreshCw size={14} /> Run Checks</button>
        </div>
      </div>

      <div className={`aurelius-card flex items-center gap-4 ${overallConfig.bg} border ${overallConfig.border}`}>
        <div className={`w-12 h-12 rounded-full ${overallConfig.bg} ${overallConfig.color} border ${overallConfig.border} flex items-center justify-center`}><OverallIcon size={24} /></div>
        <div>
          <p className={`text-xl font-bold ${overallConfig.color}`}>{overallConfig.label}</p>
          <p className="text-xs text-[#9e9eb0]">{checks.length} checks · {healthyCount} healthy · {warningCount} warning · {criticalCount} critical</p>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {checks.map((check) => {
          const cfg = statusConfig[check.status as keyof typeof statusConfig];
          const Icon = checkIcons[check.name as keyof typeof checkIcons] || Server;
          const CheckIcon = cfg.icon;
          return (
            <div key={check.name} className={`aurelius-card space-y-3 ${cfg.bg} border ${cfg.border}`}>
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-2"><Icon size={16} className={cfg.color} /><span className="text-sm font-semibold text-[#e0e0e0] capitalize">{check.name}</span></div>
                <CheckIcon size={16} className={cfg.color} />
              </div>
              <p className="text-sm text-[#9e9eb0]">{check.message}</p>
              <div className="flex items-center justify-between text-[10px] text-[#9e9eb0]">
                <span>{check.latency_ms > 0 ? `${check.latency_ms}ms` : '—'}</span>
                <span>{new Date(check.lastChecked).toLocaleTimeString()}</span>
              </div>
            </div>
          );
        })}
      </div>

      {loading && checks.length === 0 && (
        <div className="aurelius-card text-center py-12 text-[#9e9eb0]"><Loader2 size={32} className="mx-auto mb-3 animate-spin opacity-60" /><p>Running health checks...</p></div>
      )}
    </div>
  );
}
