import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Heart, Server, Activity, Database, Cpu, Wifi,
  CheckCircle, XCircle, AlertTriangle, Loader2, RefreshCw,
} from 'lucide-react';
import { useApi } from '../hooks/useApi';
import StatsCard from '../components/StatsCard';

interface HealthStatus {
  status: string; uptime: number; version: string;
  services: Array<{ name: string; status: string; latency: number; message?: string }>;
  system: { cpu: number; memory: number; disk: number };
}

export default function HealthCheckPage() {
  const { data, loading, refresh } = useApi<HealthStatus>('/health', { refreshInterval: 10000 });

  if (loading && !data) return (
    <div className="flex justify-center py-20"><Loader2 size={32} className="animate-spin text-[#4fc3f7]" /></div>
  );

  const health = data || { status: 'unknown', uptime: 0, version: '?', services: [], system: { cpu: 0, memory: 0, disk: 0 } };
  const allHealthy = health.services.every(s => s.status === 'healthy');
  const uptimeStr = health.uptime > 86400 ? `${Math.floor(health.uptime / 86400)}d` :
    health.uptime > 3600 ? `${Math.floor(health.uptime / 3600)}h` : `${Math.floor(health.uptime / 60)}m`;

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
          <Heart size={20} className="text-[#4fc3f7]" />
          System Health
        </h2>
        <button onClick={refresh} className="aurelius-btn-outline flex items-center gap-2 text-sm"><RefreshCw size={14} /> Refresh</button>
      </div>

      <div className={`p-4 rounded-xl border ${allHealthy ? 'bg-emerald-500/5 border-emerald-500/20' : 'bg-rose-500/5 border-rose-500/20'}`}>
        <div className="flex items-center gap-3">
          {allHealthy ? <CheckCircle size={24} className="text-emerald-400" /> : <AlertTriangle size={24} className="text-rose-400" />}
          <div>
            <p className={`font-bold ${allHealthy ? 'text-emerald-400' : 'text-rose-400'}`}>
              {allHealthy ? 'All Systems Operational' : 'Service Degradation'}
            </p>
            <p className="text-xs text-[#9e9eb0]">v{health.version} · Uptime: {uptimeStr} · {health.services.length} services</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3">
        <StatsCard label="CPU" value={`${health.system.cpu}%`} icon={Cpu} color={health.system.cpu > 80 ? 'text-rose-400' : 'text-emerald-400'} />
        <StatsCard label="Memory" value={`${health.system.memory}%`} icon={Database} color={health.system.memory > 80 ? 'text-rose-400' : 'text-emerald-400'} />
        <StatsCard label="Disk" value={`${health.system.disk}%`} icon={Server} color={health.system.disk > 80 ? 'text-rose-400' : 'text-emerald-400'} />
      </div>

      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-[#e0e0e0]">Services</h3>
        {health.services.map(service => (
          <motion.div key={service.name} initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            className="aurelius-card p-4 flex items-center justify-between"
          >
            <div className="flex items-center gap-3">
              {service.status === 'healthy' ? <CheckCircle size={16} className="text-emerald-400" /> :
               service.status === 'degraded' ? <AlertTriangle size={16} className="text-amber-400" /> :
               <XCircle size={16} className="text-rose-400" />}
              <div>
                <p className="text-sm font-medium text-[#e0e0e0]">{service.name}</p>
                {service.message && <p className="text-xs text-[#9e9eb0]">{service.message}</p>}
              </div>
            </div>
            <div className="text-right">
              <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${
                service.status === 'healthy' ? 'text-emerald-400 bg-emerald-500/10' :
                service.status === 'degraded' ? 'text-amber-400 bg-amber-500/10' :
                'text-rose-400 bg-rose-500/10'
              }`}>{service.status}</span>
              <p className="text-[10px] text-[#9e9eb0] mt-0.5">{service.latency}ms</p>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  );
}
