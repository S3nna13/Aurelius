import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Cpu, ArrowLeft, Send } from 'lucide-react';
import Input from '../components/ui/Input';
import Select from '../components/ui/Select';
import Textarea from '../components/ui/Textarea';

const CAPABILITIES = [
  'code', 'review', 'debug', 'research', 'analyze',
  'summarize', 'search', 'test', 'refactor', 'monitor',
];

const ROLES = ['worker', 'specialist', 'supervisor', 'orchestrator'];

export default function AgentSpawnForm() {
  const navigate = useNavigate();
  const [name, setName] = useState('');
  const [role, setRole] = useState('worker');
  const [capabilities, setCapabilities] = useState<string[]>([]);
  const [description, setDescription] = useState('');
  const [spawning, setSpawning] = useState(false);
  const [error, setError] = useState('');

  const toggleCapability = (cap: string) => {
    setCapabilities(prev =>
      prev.includes(cap) ? prev.filter(c => c !== cap) : [...prev, cap]
    );
  };

  const handleSubmit = async () => {
    if (!name.trim()) { setError('Name is required'); return; }
    setSpawning(true);
    setError('');
    try {
      const res = await fetch('/api/agents', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name.trim(), role, capabilities }),
      });
      if (!res.ok) { const d = await res.json(); throw new Error(d.error || 'Failed to spawn'); }
      const { agent } = await res.json();
      navigate(`/agents/${agent.id}`);
    } catch (e: any) {
      setError(e.message);
    }
    setSpawning(false);
  };

  return (
    <div className="max-w-2xl mx-auto space-y-6">
      <div className="flex items-center gap-3">
        <button onClick={() => navigate('/agents')} className="text-[#9e9eb0] hover:text-[#e0e0e0] transition-colors">
          <ArrowLeft size={20} />
        </button>
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
          <Cpu size={20} className="text-[#4fc3f7]" />
          Spawn Agent
        </h2>
      </div>

      <div className="aurelius-card p-6 space-y-5">
        <div>
          <label className="block text-sm font-medium text-[#e0e0e0] mb-1">Agent Name</label>
          <Input value={name} onChange={e => setName(e.target.value)} placeholder="MyCodingAgent" />
        </div>

        <div>
          <label className="block text-sm font-medium text-[#e0e0e0] mb-1">Role</label>
          <Select value={role} onChange={e => setRole(e.target.value)}>
            {ROLES.map(r => <option key={r} value={r}>{r}</option>)}
          </Select>
        </div>

        <div>
          <label className="block text-sm font-medium text-[#e0e0e0] mb-2">Capabilities</label>
          <div className="flex flex-wrap gap-2">
            {CAPABILITIES.map(cap => (
              <button
                key={cap}
                onClick={() => toggleCapability(cap)}
                className={`px-3 py-1.5 text-xs rounded-lg border transition-colors ${
                  capabilities.includes(cap)
                    ? 'bg-[#4fc3f7]/10 border-[#4fc3f7]/40 text-[#4fc3f7]'
                    : 'bg-[#0f0f1a] border-[#2d2d44] text-[#9e9eb0] hover:text-[#e0e0e0]'
                }`}
              >
                {cap}
              </button>
            ))}
          </div>
        </div>

        <div>
          <label className="block text-sm font-medium text-[#e0e0e0] mb-1">Description</label>
          <Textarea value={description} onChange={e => setDescription(e.target.value)} placeholder="Optional agent description..." rows={3} />
        </div>

        {error && <p className="text-rose-400 text-sm">{error}</p>}

        <button
          onClick={handleSubmit}
          disabled={spawning}
          className="aurelius-btn-primary w-full flex items-center justify-center gap-2"
        >
          {spawning ? <span className="animate-pulse">Spawning...</span> : <><Send size={16} /> Spawn Agent</>}
        </button>
      </div>
    </div>
  );
}
