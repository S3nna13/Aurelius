import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  Wrench, Search, Plus, Trash2, Play, CheckCircle, XCircle,
  Code, Terminal, Database, Globe, Mail, Clock,
} from 'lucide-react';
import Input from '../components/ui/Input';
import EmptyState from '../components/EmptyState';

interface Tool {
  id: string; name: string; description: string;
  category: string; enabled: boolean; lastUsed?: string;
  usageCount: number; schema?: Record<string, unknown>;
}

const DEFAULT_TOOLS: Tool[] = [
  { id: 'read_file', name: 'Read File', description: 'Read contents of a file from the filesystem.', category: 'filesystem', enabled: true, usageCount: 142, lastUsed: '2 min ago' },
  { id: 'write_file', name: 'Write File', description: 'Write content to a file.', category: 'filesystem', enabled: true, usageCount: 98, lastUsed: '5 min ago' },
  { id: 'search_web', name: 'Web Search', description: 'Search the web for information.', category: 'web', enabled: true, usageCount: 76, lastUsed: '10 min ago' },
  { id: 'run_command', name: 'Run Command', description: 'Execute a shell command.', category: 'system', enabled: false, usageCount: 34, lastUsed: '1 hour ago' },
  { id: 'query_db', name: 'Query Database', description: 'Run SQL queries against connected databases.', category: 'data', enabled: true, usageCount: 28, lastUsed: '30 min ago' },
  { id: 'send_email', name: 'Send Email', description: 'Send an email via configured SMTP.', category: 'communication', enabled: false, usageCount: 12, lastUsed: '2 days ago' },
];

const CATEGORY_ICONS: Record<string, typeof Terminal> = {
  filesystem: Terminal, web: Globe, system: Terminal, data: Database, communication: Mail,
};

export default function AgentToolsRegistry() {
  const [search, setSearch] = useState('');
  const [category, setCategory] = useState('all');
  const [tools, setTools] = useState<Tool[]>(DEFAULT_TOOLS);
  const [selectedTool, setSelectedTool] = useState<Tool | null>(null);

  const filtered = tools.filter(t => {
    if (category !== 'all' && t.category !== category) return false;
    return !search || t.name.toLowerCase().includes(search.toLowerCase()) || t.description.toLowerCase().includes(search.toLowerCase());
  });

  const toggleTool = (id: string) => {
    setTools(prev => prev.map(t => t.id === id ? { ...t, enabled: !t.enabled } : t));
  };

  const categories = ['all', ...new Set(tools.map(t => t.category))];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
          <Wrench size={20} className="text-[#4fc3f7]" /> Tool Registry
          <span className="text-sm font-normal text-[#9e9eb0]">({tools.filter(t => t.enabled).length} active)</span>
        </h2>
        <div className="flex gap-2">
          <div className="relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#9e9eb0]" />
            <Input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search tools..." className="pl-8 py-1.5 text-sm w-48" />
          </div>
          <select value={category} onChange={e => setCategory(e.target.value)}
            className="bg-[#0f0f1a] border border-[#2d2d44] rounded-lg px-3 py-1.5 text-sm text-[#e0e0e0]">
            {categories.map(c => <option key={c} value={c}>{c === 'all' ? 'All' : c}</option>)}
          </select>
        </div>
      </div>

      {filtered.length === 0 && <EmptyState icon={Wrench} title="No Tools Found" description="Try adjusting your search or register a new tool." />}

      <div className="grid grid-cols-1 gap-2">
        {filtered.map(tool => {
          const Icon = CATEGORY_ICONS[tool.category] || Terminal;
          return (
            <motion.div key={tool.id} layout initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              className="aurelius-card p-4 flex items-center justify-between hover:border-[#4fc3f7]/30 transition-all cursor-pointer"
              onClick={() => setSelectedTool(tool)}
            >
              <div className="flex items-center gap-3 min-w-0">
                <div className={`w-9 h-9 rounded-lg flex items-center justify-center ${tool.enabled ? 'bg-emerald-500/10' : 'bg-[#0f0f1a]'}`}>
                  <Icon size={16} className={tool.enabled ? 'text-emerald-400' : 'text-[#9e9eb0]'} />
                </div>
                <div className="min-w-0">
                  <h3 className="text-sm font-medium text-[#e0e0e0]">{tool.name}</h3>
                  <p className="text-xs text-[#9e9eb0] truncate">{tool.description}</p>
                </div>
              </div>
              <div className="flex items-center gap-4 shrink-0">
                <span className="text-[10px] text-[#9e9eb0]">{tool.usageCount} calls</span>
                <label className="relative inline-flex items-center cursor-pointer" onClick={e => e.stopPropagation()}>
                  <input type="checkbox" checked={tool.enabled} onChange={() => toggleTool(tool.id)} className="sr-only peer" />
                  <div className="w-8 h-4 bg-[#2d2d44] rounded-full peer peer-checked:bg-emerald-500/30 after:content-[''] after:absolute after:top-0.5 after:left-0.5 after:bg-[#9e9eb0] after:rounded-full after:h-3 after:w-3 after:transition-all peer-checked:after:translate-x-4 peer-checked:after:bg-emerald-400" />
                </label>
              </div>
            </motion.div>
          );
        })}
      </div>

      {selectedTool && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4" onClick={() => setSelectedTool(null)}>
          <div className="aurelius-card p-6 max-w-lg w-full space-y-4" onClick={e => e.stopPropagation()}>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-lg bg-[#4fc3f7]/10 flex items-center justify-center"><Wrench size={20} className="text-[#4fc3f7]" /></div>
              <div>
                <h3 className="text-lg font-bold text-[#e0e0e0]">{selectedTool.name}</h3>
                <p className="text-xs text-[#9e9eb0]">{selectedTool.category}</p>
              </div>
            </div>
            <p className="text-sm text-[#9e9eb0]">{selectedTool.description}</p>
            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="bg-[#0f0f1a] p-3 rounded-lg"><span className="text-[#9e9eb0]">Status</span><p className="text-[#e0e0e0] font-medium mt-0.5">{selectedTool.enabled ? 'Active' : 'Disabled'}</p></div>
              <div className="bg-[#0f0f1a] p-3 rounded-lg"><span className="text-[#9e9eb0]">Total Calls</span><p className="text-[#e0e0e0] font-medium mt-0.5">{selectedTool.usageCount}</p></div>
              {selectedTool.lastUsed && <div className="bg-[#0f0f1a] p-3 rounded-lg"><span className="text-[#9e9eb0]">Last Used</span><p className="text-[#e0e0e0] font-medium mt-0.5">{selectedTool.lastUsed}</p></div>}
            </div>
            <button onClick={() => toggleTool(selectedTool.id)} className={`w-full py-2 rounded-lg text-sm font-medium transition-colors ${selectedTool.enabled ? 'bg-rose-500/10 text-rose-400 hover:bg-rose-500/20' : 'bg-emerald-500/10 text-emerald-400 hover:bg-emerald-500/20'}`}>
              {selectedTool.enabled ? 'Disable Tool' : 'Enable Tool'}
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
