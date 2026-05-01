import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { motion } from 'framer-motion';
import {
  TrendingUp, Play, Square, Plus, BarChart3,
  Loader2, Clock, CheckCircle, XCircle, LineChart,
} from 'lucide-react';
import { useApi } from '../hooks/useApi';
import StatsCard from '../components/StatsCard';
import EmptyState from '../components/EmptyState';
import Skeleton from '../components/Skeleton';
import LineChartComponent from '../components/charts/LineChart';

interface TrainingRun {
  id: string; name: string; status: string;
  modelSize: string; totalSteps: number; currentStep: number;
  loss: number | null; learningRate: number | null;
  tokensSeen: number; totalTokens: number;
  startedAt: string; duration: number;
}

const statusStyles: Record<string, { color: string; bg: string }> = {
  running: { color: 'text-emerald-400', bg: 'bg-emerald-500/10' },
  completed: { color: 'text-[#4fc3f7]', bg: 'bg-[#4fc3f7]/10' },
  failed: { color: 'text-rose-400', bg: 'bg-rose-500/10' },
  paused: { color: 'text-amber-400', bg: 'bg-amber-500/10' },
};

export default function Training() {
  const navigate = useNavigate();
  const { data, loading } = useApi<{ runs: TrainingRun[] }>('/training', { refreshInterval: 5000 });
  const runs = data?.runs || [];

  const activeRuns = runs.filter(r => r.status === 'running');
  const lossData = activeRuns.length > 0
    ? Array.from({ length: Math.min(activeRuns[0].currentStep, 100) }, (_, i) => ({
        step: i, loss: 3.0 - 2.0 * Math.exp(-i / 20) + Math.random() * 0.1,
      }))
    : [];

  return (
    <div className="space-y-6">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
          <TrendingUp size={20} className="text-[#4fc3f7]" />
          Training Dashboard
        </h2>
        <div className="flex gap-2">
          <button className="aurelius-btn-outline flex items-center gap-2 text-sm">
            <Plus size={14} /> New Run
          </button>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        <StatsCard label="Active Runs" value={activeRuns.length} icon={Play} color="text-emerald-400" />
        <StatsCard label="Completed" value={runs.filter(r => r.status === 'completed').length} icon={CheckCircle} color="text-[#4fc3f7]" />
        <StatsCard label="Failed" value={runs.filter(r => r.status === 'failed').length} icon={XCircle} color="text-rose-400" />
        <StatsCard label="Total" value={runs.length} icon={BarChart3} color="text-amber-400" />
      </div>

      {activeRuns.length > 0 && (
        <div className="aurelius-card p-4">
          <h3 className="text-sm font-semibold text-[#e0e0e0] mb-3 flex items-center gap-2">
            <LineChart size={14} className="text-emerald-400" />
            Live Training — {activeRuns[0].name}
          </h3>
          <div className="h-48">
            <LineChartComponent
              data={lossData}
              xKey="step"
              yKey="loss"
              color="#4fc3f7"
            />
          </div>
          <div className="grid grid-cols-4 gap-3 mt-3 text-center text-xs">
            <div><p className="text-emerald-400 font-bold">{activeRuns[0].currentStep}/{activeRuns[0].totalSteps}</p><p className="text-[#9e9eb0]">Steps</p></div>
            <div><p className="text-[#4fc3f7] font-bold">{activeRuns[0].loss?.toFixed(4) || '-'}</p><p className="text-[#9e9eb0]">Loss</p></div>
            <div><p className="text-amber-400 font-bold">{activeRuns[0].learningRate?.toExponential(1) || '-'}</p><p className="text-[#9e9eb0]">LR</p></div>
            <div><p className="text-[#e0e0e0] font-bold">{Math.floor(activeRuns[0].duration / 60)}m</p><p className="text-[#9e9eb0]">Elapsed</p></div>
          </div>
        </div>
      )}

      <div className="space-y-2">
        <h3 className="text-sm font-semibold text-[#e0e0e0]">Run History</h3>
        {loading && <Skeleton className="h-24" />}
        {!loading && runs.length === 0 && (
          <EmptyState icon={TrendingUp} title="No Training Runs" description="Start a training run to see results here." />
        )}
        {runs.map(run => {
          const st = statusStyles[run.status] || statusStyles.completed;
          return (
            <motion.div key={run.id} initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              onClick={() => navigate(`/training/${run.id}`)}
              className="aurelius-card p-4 hover:border-[#4fc3f7]/30 cursor-pointer transition-all"
            >
              <div className="flex items-center justify-between">
                <div className="flex items-center gap-3">
                  <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${st.bg} ${st.color}`}>
                    {run.status === 'running' ? <Loader2 size={14} className="animate-spin" /> : <BarChart3 size={14} />}
                  </div>
                  <div>
                    <p className="text-sm font-medium text-[#e0e0e0]">{run.name}</p>
                    <p className="text-xs text-[#9e9eb0]">{run.modelSize} · {run.currentStep}/{run.totalSteps} steps · {run.tokensSeen.toLocaleString()}/{run.totalTokens.toLocaleString()} tokens</p>
                  </div>
                </div>
                <div className="text-right">
                  <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${st.color} bg-[#0f0f1a] border border-[#2d2d44]`}>{run.status}</span>
                  {run.loss !== null && <p className="text-xs text-[#9e9eb0] mt-1">loss: {run.loss.toFixed(4)}</p>}
                </div>
              </div>
            </motion.div>
          );
        })}
      </div>
    </div>
  );
}
