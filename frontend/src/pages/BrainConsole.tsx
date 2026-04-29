import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Brain, Send, Loader2, Lightbulb, GitBranch,
  CheckCircle, AlertCircle, Cpu, Activity,
} from 'lucide-react';

export default function BrainConsole() {
  const [input, setInput] = useState('');
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [upgradeResult, setUpgradeResult] = useState<any>(null);
  const [upgradeLoading, setUpgradeLoading] = useState(false);

  const think = async () => {
    if (!input.trim()) return;
    setLoading(true);
    try {
      const res = await fetch('/api/brain/think', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ input }),
      });
      setResult(await res.json());
    } catch { setResult({ error: 'Brain call failed' }); }
    setLoading(false);
  };

  const runUpgrade = async () => {
    setUpgradeLoading(true);
    try {
      const res = await fetch('/api/brain/upgrade/run', { method: 'POST' });
      setUpgradeResult(await res.json());
    } catch { setUpgradeResult({ error: 'Upgrade failed' }); }
    setUpgradeLoading(false);
  };

  return (
    <div className="space-y-6">
      <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
        <Brain size={20} className="text-[#4fc3f7]" /> Neural Brain Console
      </h2>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-4">
        <div className="aurelius-card p-4 space-y-4">
          <h3 className="text-sm font-semibold text-[#e0e0e0] flex items-center gap-2">
            <Lightbulb size={14} className="text-[#4fc3f7]" /> Think
          </h3>
          <textarea value={input} onChange={e => setInput(e.target.value)}
            placeholder="Enter a task for the neural brain..."
            rows={4}
            className="w-full bg-[#0f0f1a] border border-[#2d2d44] rounded-lg p-3 text-sm text-[#e0e0e0] placeholder:text-[#9e9eb0] resize-none" />
          <button onClick={think} disabled={loading || !input.trim()}
            className="aurelius-btn-primary w-full flex items-center justify-center gap-2 disabled:opacity-50">
            {loading ? <Loader2 size={14} className="animate-spin" /> : <Brain size={14} />}
            {loading ? 'Thinking...' : 'Run Neural Brain'}
          </button>
          {result && (
            <div className="bg-[#0a0a14] rounded-lg p-3 text-xs font-mono space-y-2 max-h-80 overflow-y-auto">
              {result.error ? (
                <p className="text-rose-400">{result.error}</p>
              ) : (
                <>
                  <div className="flex gap-2">
                    <span className="text-emerald-400">State:</span>
                    <span className="text-[#e0e0e0]">{result.state}</span>
                  </div>
                  <div className="flex gap-2">
                    <span className="text-[#4fc3f7]">Plan:</span>
                    <span className="text-[#e0e0e0]">{result.plan?.length || 0} steps</span>
                  </div>
                  <div className="flex gap-2">
                    <span className="text-amber-400">Reasoning:</span>
                    <span className="text-[#e0e0e0]">{result.reasoning_steps} steps</span>
                  </div>
                  <div className="flex gap-2">
                    <span className="text-violet-400">Actions:</span>
                    <span className="text-[#e0e0e0]">{result.actions} taken</span>
                  </div>
                  <div className="flex gap-2">
                    <span className="text-rose-400">Reflections:</span>
                    <span className="text-[#e0e0e0]">{result.reflections}</span>
                  </div>
                  {result.output && (
                    <div className="pt-2 border-t border-[#2d2d44]">
                      <p className="text-[#9e9eb0] mb-1">Output:</p>
                      <p className="text-[#e0e0e0] whitespace-pre-wrap">{result.output}</p>
                    </div>
                  )}
                </>
              )}
            </div>
          )}
        </div>

        <div className="aurelius-card p-4 space-y-4">
          <h3 className="text-sm font-semibold text-[#e0e0e0] flex items-center gap-2">
            <GitBranch size={14} className="text-emerald-400" /> Self-Upgrade
          </h3>
          <p className="text-xs text-[#9e9eb0]">Run an automatic upgrade cycle. The system will observe metrics, research improvements, generate code, test, safety-check, and deploy.</p>
          <button onClick={runUpgrade} disabled={upgradeLoading}
            className="aurelius-btn-primary w-full flex items-center justify-center gap-2 disabled:opacity-50">
            {upgradeLoading ? <Loader2 size={14} className="animate-spin" /> : <GitBranch size={14} />}
            {upgradeLoading ? 'Upgrading...' : 'Run Upgrade Cycle'}
          </button>
          {upgradeResult && (
            <div className="bg-[#0a0a14] rounded-lg p-3 text-xs font-mono space-y-1">
              {upgradeResult.error ? (
                <p className="text-rose-400">{upgradeResult.error}</p>
              ) : (
                <>
                  <p className="text-[#9e9eb0]">System: Self-Upgrade Layer</p>
                  <p className="text-[#9e9eb0]">Status: <span className="text-emerald-400">{upgradeResult.state}</span></p>
                  <p className="text-[#9e9eb0]">Improvements: <span className="text-[#4fc3f7]">{upgradeResult.improvements}</span></p>
                  <p className="text-[#9e9eb0]">Failures: <span className="text-rose-400">{upgradeResult.failures}</span></p>
                  <p className="text-[#9e9eb0]">Cycles: <span className="text-[#e0e0e0]">{upgradeResult.cycles}</span></p>
                </>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
