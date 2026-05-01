import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  GitBranch, Play, Square, Plus, Trash2, GripVertical,
  CheckCircle, XCircle, Clock, Loader2, Settings,
} from 'lucide-react';
import { useApi } from '../hooks/useApi';
import EmptyState from '../components/EmptyState';
import Skeleton from '../components/Skeleton';
import Modal from '../components/ui/Modal';
import Input from '../components/ui/Input';
import Select from '../components/ui/Select';

interface WorkflowStep {
  id: string; name: string; type: string;
  config: Record<string, unknown>;
}

interface Workflow {
  id: string; name: string; description: string;
  steps: WorkflowStep[]; enabled: boolean;
  lastRun?: string; lastStatus?: string;
}

const STEP_TYPES = [
  { id: 'agent_task', name: 'Agent Task', color: 'text-[#4fc3f7]' },
  { id: 'tool_call', name: 'Tool Call', color: 'text-amber-400' },
  { id: 'condition', name: 'Condition', color: 'text-violet-400' },
  { id: 'delay', name: 'Delay', color: 'text-gray-400' },
  { id: 'notification', name: 'Notification', color: 'text-emerald-400' },
];

export default function Workflows() {
  const { data, loading } = useApi<{ workflows: Workflow[] }>('/workflows', { refreshInterval: 10000 });
  const [selected, setSelected] = useState<Workflow | null>(null);
  const [showNew, setShowNew] = useState(false);
  const [newName, setNewName] = useState('');
  const workflows = data?.workflows || [];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
          <GitBranch size={20} className="text-[#4fc3f7]" />
          Workflows
        </h2>
        <button onClick={() => setShowNew(true)} className="aurelius-btn-primary flex items-center gap-2 text-sm">
          <Plus size={14} /> New Workflow
        </button>
      </div>

      {loading && <Skeleton className="h-32" />}

      {!loading && workflows.length === 0 && (
        <EmptyState icon={GitBranch} title="No Workflows" description="Create automated workflows to chain agent tasks together." />
      )}

      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {workflows.map(wf => (
          <motion.div key={wf.id} initial={{ opacity: 0 }} animate={{ opacity: 1 }}
            onClick={() => setSelected(wf)}
            className="aurelius-card p-4 hover:border-[#4fc3f7]/30 cursor-pointer transition-all"
          >
            <div className="flex items-start justify-between mb-3">
              <div>
                <h3 className="text-sm font-semibold text-[#e0e0e0]">{wf.name}</h3>
                <p className="text-xs text-[#9e9eb0] mt-0.5">{wf.description}</p>
              </div>
              <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${wf.enabled ? 'text-emerald-400 bg-emerald-500/10 border border-emerald-500/20' : 'text-[#9e9eb0] bg-[#0f0f1a] border border-[#2d2d44]'}`}>
                {wf.enabled ? 'Active' : 'Disabled'}
              </span>
            </div>
            <div className="flex items-center gap-1.5 flex-wrap">
              {wf.steps.slice(0, 5).map((step, i) => (
                <div key={step.id} className="flex items-center gap-1">
                  <span className="text-[10px] bg-[#0f0f1a] border border-[#2d2d44] px-2 py-0.5 rounded text-[#9e9eb0]">{step.name}</span>
                  {i < wf.steps.length - 1 && <span className="text-[#2d2d44] text-xs">→</span>}
                </div>
              ))}
              {wf.steps.length > 5 && <span className="text-[10px] text-[#9e9eb0]">+{wf.steps.length - 5} more</span>}
            </div>
            {wf.lastRun && (
              <div className="flex items-center gap-2 mt-3 text-[10px] text-[#9e9eb0]">
                <span>Last run: {wf.lastRun}</span>
                {wf.lastStatus === 'completed' && <CheckCircle size={10} className="text-emerald-400" />}
                {wf.lastStatus === 'failed' && <XCircle size={10} className="text-rose-400" />}
              </div>
            )}
          </motion.div>
        ))}
      </div>

      <AnimatePresence>
        {showNew && (
          <Modal onClose={() => setShowNew(false)}>
            <div className="space-y-4">
              <h3 className="text-lg font-bold text-[#e0e0e0]">New Workflow</h3>
              <Input value={newName} onChange={e => setNewName(e.target.value)} placeholder="Workflow name" />
              <textarea placeholder="Description" className="w-full bg-[#0f0f1a] border border-[#2d2d44] rounded-lg p-3 text-sm text-[#e0e0e0] h-24 resize-none" />
              <button className="aurelius-btn-primary w-full">Create Workflow</button>
            </div>
          </Modal>
        )}
      </AnimatePresence>
    </div>
  );
}
