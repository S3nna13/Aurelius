import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Puzzle, Plus, Save, Trash2, Code, Eye, EyeOff,
  ChevronRight, GripVertical, Play, CheckCircle, X,
} from 'lucide-react';
import Input from '../components/ui/Input';
import Select from '../components/ui/Select';
import Toggle from '../components/ui/Toggle';
import { useToast } from '../components/ToastProvider';

interface SkillStep {
  id: string; type: string; config: string; description: string;
}

const STEP_TYPES = [
  { id: 'prompt', label: 'Prompt Template', color: 'text-[#4fc3f7]' },
  { id: 'tool_call', label: 'Tool Call', color: 'text-amber-400' },
  { id: 'condition', label: 'Condition', color: 'text-violet-400' },
  { id: 'loop', label: 'Loop/Repeat', color: 'text-emerald-400' },
  { id: 'output', label: 'Output Format', color: 'text-rose-400' },
];

export default function AgentSkillBuilder() {
  const { toast } = useToast();
  const [name, setName] = useState('');
  const [description, setDescription] = useState('');
  const [category, setCategory] = useState('custom');
  const [steps, setSteps] = useState<SkillStep[]>([]);
  const [editingStep, setEditingStep] = useState<number | null>(null);
  const [showCode, setShowCode] = useState(false);

  const addStep = () => {
    const step: SkillStep = {
      id: Date.now().toString(36),
      type: 'prompt',
      config: '{}',
      description: '',
    };
    setSteps(prev => [...prev, step]);
    setEditingStep(steps.length);
  };

  const removeStep = (idx: number) => {
    setSteps(prev => prev.filter((_, i) => i !== idx));
    setEditingStep(null);
  };

  const moveStep = (from: number, to: number) => {
    if (to < 0 || to >= steps.length) return;
    const copy = [...steps];
    const [item] = copy.splice(from, 1);
    copy.splice(to, 0, item);
    setSteps(copy);
  };

  const handleSave = () => {
    if (!name.trim()) { toast('Skill name required', 'error'); return; }
    const skill = { name, description, category, steps };
    localStorage.setItem(`skill_${name}`, JSON.stringify(skill));
    toast('Skill saved', 'success');
  };

  const skillJson = {
    name: name || 'unnamed_skill',
    description,
    category,
    steps: steps.map(s => ({ type: s.type, config: JSON.parse(s.config || '{}'), description: s.description })),
  };

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
        <Puzzle size={20} className="text-[#4fc3f7]" /> Custom Skill Builder
      </h2>

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 space-y-4">
          <div className="aurelius-card p-4 space-y-4">
            <Input value={name} onChange={e => setName(e.target.value)} placeholder="Skill name" />
            <textarea value={description} onChange={e => setDescription(e.target.value)}
              placeholder="Describe what this skill does..." rows={3}
              className="w-full bg-[#0f0f1a] border border-[#2d2d44] rounded-lg p-3 text-sm text-[#e0e0e0] placeholder:text-[#9e9eb0] resize-none" />
            <Select value={category} onChange={e => setCategory(e.target.value)}>
              <option value="custom">Custom</option>
              <option value="coding">Coding</option>
              <option value="research">Research</option>
              <option value="analysis">Analysis</option>
              <option value="automation">Automation</option>
            </Select>
          </div>

          <div className="aurelius-card p-4">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-sm font-semibold text-[#e0e0e0]">Steps ({steps.length})</h3>
              <button onClick={addStep} className="aurelius-btn-primary flex items-center gap-1.5 text-xs">
                <Plus size={12} /> Add Step
              </button>
            </div>

            {steps.length === 0 && (
              <div className="text-center py-8 text-[#9e9eb0] text-sm">
                No steps yet. Click "Add Step" to build your skill workflow.
              </div>
            )}

            <div className="space-y-2">
              {steps.map((step, idx) => {
                const stepType = STEP_TYPES.find(t => t.id === step.type);
                const isEditing = editingStep === idx;
                return (
                  <motion.div key={step.id} layout initial={{ opacity: 0, x: -20 }} animate={{ opacity: 1, x: 0 }}
                    className="border border-[#2d2d44] rounded-lg overflow-hidden"
                  >
                    <div className="flex items-center gap-2 p-3 bg-[#0f0f1a] cursor-pointer" onClick={() => setEditingStep(isEditing ? null : idx)}>
                      <GripVertical size={14} className="text-[#2d2d44] cursor-grab" />
                      <span className={`text-[10px] font-bold px-2 py-0.5 rounded-full ${stepType?.color || 'text-gray-400'} bg-[#0a0a14]`}>
                        {stepType?.label || step.type}
                      </span>
                      <span className="text-xs text-[#9e9eb0] flex-1 truncate">{step.description || 'No description'}</span>
                      <button onClick={e => { e.stopPropagation(); removeStep(idx); }} className="text-rose-400/60 hover:text-rose-400">
                        <Trash2 size={12} />
                      </button>
                    </div>
                    {isEditing && (
                      <div className="p-3 border-t border-[#2d2d44] space-y-2">
                        <Select value={step.type} onChange={e => setSteps(prev => prev.map((s, i) => i === idx ? { ...s, type: e.target.value } : s))}>
                          {STEP_TYPES.map(t => <option key={t.id} value={t.id}>{t.label}</option>)}
                        </Select>
                        <Input value={step.description} onChange={e => setSteps(prev => prev.map((s, i) => i === idx ? { ...s, description: e.target.value } : s))} placeholder="Step description" />
                        <textarea value={step.config} onChange={e => setSteps(prev => prev.map((s, i) => i === idx ? { ...s, config: e.target.value } : s))}
                          placeholder='{"key": "value"}' rows={3}
                          className="w-full bg-[#0a0a14] border border-[#2d2d44] rounded p-2 text-xs text-[#4fc3f7] font-mono" />
                      </div>
                    )}
                  </motion.div>
                );
              })}
            </div>

            {steps.length > 1 && (
              <div className="flex items-center gap-2 mt-3 text-xs text-[#9e9eb0]">
                <span className="flex items-center gap-1">{steps[0]?.description || 'Start'}</span>
                <ChevronRight size={10} />
                <span className="flex items-center gap-1">{steps[steps.length - 1]?.description || 'End'}</span>
              </div>
            )}
          </div>

          <button onClick={handleSave} className="aurelius-btn-primary w-full flex items-center justify-center gap-2">
            <Save size={14} /> Save Skill
          </button>
        </div>

        <div className="space-y-4">
          <div className="aurelius-card p-4">
            <div className="flex items-center justify-between mb-3">
              <h3 className="text-xs font-bold text-[#e0e0e0] uppercase tracking-wider">Preview</h3>
              <button onClick={() => setShowCode(!showCode)} className="text-[#9e9eb0] hover:text-[#e0e0e0]">
                {showCode ? <EyeOff size={14} /> : <Eye size={14} />}
              </button>
            </div>
            {showCode ? (
              <pre className="text-[10px] text-[#4fc3f7]/80 bg-[#0a0a14] p-3 rounded-lg overflow-x-auto max-h-96 font-mono">
                {JSON.stringify(skillJson, null, 2)}
              </pre>
            ) : (
              <div className="space-y-2">
                <div className="flex items-center gap-2 text-xs text-[#9e9eb0]">
                  <Puzzle size={12} className="text-[#4fc3f7]" /> {name || 'unnamed_skill'}
                </div>
                <div className="flex items-center gap-2 text-xs text-[#9e9eb0]">
                  <span>Category:</span> <span className="text-[#e0e0e0]">{category}</span>
                </div>
                <div className="flex items-center gap-2 text-xs text-[#9e9eb0]">
                  <span>Steps:</span> <span className="text-[#e0e0e0]">{steps.length}</span>
                </div>
                <div className="flex items-center gap-2 text-xs text-[#9e9eb0]">
                  <span>Types:</span>
                  <div className="flex gap-1">
                    {[...new Set(steps.map(s => s.type))].map(t => {
                      const st = STEP_TYPES.find(step => step.id === t);
                      return <span key={t} className={`text-[10px] px-1.5 py-0.5 rounded ${st?.color || 'text-gray-400'} bg-[#0a0a14]`}>{t}</span>;
                    })}
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
