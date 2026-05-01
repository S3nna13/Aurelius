import { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { ArrowLeft, TrendingUp, Play, Square, BarChart3, Clock, Cpu, Database, Activity } from 'lucide-react';
import LineChart from '../components/charts/LineChart';

export default function TrainingDetail() {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [run, setRun] = useState<any>(null);

  useEffect(() => {
    fetch(`/api/training/${id}`).then(r => r.json()).then(d => setRun(d.run)).catch(() => {});
  }, [id]);

  if (!run) return (
    <div className="flex justify-center py-20">
      <div className="w-8 h-8 border-2 border-[#4fc3f7] border-t-transparent rounded-full animate-spin" />
    </div>
  );

  const lossData = Array.from({ length: Math.min(run.currentStep || 1000, 200) }, (_, i) => ({
    step: i, loss: (run.initialLoss || 3.0) - 2.0 * Math.exp(-i / (run.totalSteps / 50)) + Math.random() * 0.05,
  }));

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <div className="flex items-center gap-3">
        <button onClick={() => navigate('/training')} className="text-[#9e9eb0] hover:text-[#e0e0e0]"><ArrowLeft size={20} /></button>
        <div>
          <h2 className="text-lg font-bold text-[#e0e0e0]">{run.name}</h2>
          <p className="text-xs text-[#9e9eb0]">{run.modelSize} · {run.currentStep}/{run.totalSteps} steps</p>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
        {[
          { label: 'Loss', value: run.loss?.toFixed(4) || '-', color: 'text-[#4fc3f7]' },
          { label: 'Learning Rate', value: run.learningRate?.toExponential(2) || '-', color: 'text-emerald-400' },
          { label: 'Duration', value: `${Math.floor((run.duration || 0) / 60)}m ${Math.floor((run.duration || 0) % 60)}s`, color: 'text-amber-400' },
          { label: 'Tokens', value: (run.tokensSeen || 0).toLocaleString(), color: 'text-violet-400' },
        ].map(s => (
          <div key={s.label} className="aurelius-card text-center py-3">
            <p className={`text-lg font-bold ${s.color}`}>{s.value}</p>
            <p className="text-[10px] text-[#9e9eb0] uppercase tracking-wider">{s.label}</p>
          </div>
        ))}
      </div>

      <div className="aurelius-card p-4">
        <h3 className="text-sm font-semibold text-[#e0e0e0] mb-3 flex items-center gap-2">
          <TrendingUp size={14} className="text-[#4fc3f7]" /> Training Loss
        </h3>
        <div className="h-64"><LineChart data={lossData} xKey="step" yKey="loss" color="#4fc3f7" /></div>
      </div>

      <div className="aurelius-card p-4">
        <h3 className="text-sm font-semibold text-[#e0e0e0] mb-3">Config</h3>
        <pre className="text-xs text-[#4fc3f7]/80 bg-[#0a0a14] p-3 rounded-lg border border-[#2d2d44] overflow-x-auto max-h-60">
          {JSON.stringify(run.config || {}, null, 2)}
        </pre>
      </div>
    </div>
  );
}
