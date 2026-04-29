import { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import {
  Puzzle, Search, Filter, Power, PowerOff, Plus,
  Cpu, Globe, Database, MessageSquare, Terminal, Code,
  BarChart3, Shield, Server, Calendar, GraduationCap,
  Wrench, CheckCircle, XCircle, Star,
} from 'lucide-react';
import Input from '../components/ui/Input';
import EmptyState from '../components/EmptyState';
import Skeleton from '../components/Skeleton';
import Toggle from '../components/ui/Toggle';

const PLUGIN_ICONS: Record<string, typeof Puzzle> = {
  filesystem: Terminal, web: Globe, database: Database, communication: MessageSquare,
  system: Server, code: Code, ai: Cpu, analytics: BarChart3,
  security: Shield, devops: Server, productivity: Calendar, education: GraduationCap,
};

const PLUGIN_COLORS: Record<string, string> = {
  filesystem: 'text-emerald-400', web: 'text-[#4fc3f7]', database: 'text-indigo-400',
  communication: 'text-violet-400', system: 'text-amber-400', code: 'text-cyan-400',
  ai: 'text-rose-400', analytics: 'text-lime-400', security: 'text-red-400',
  devops: 'text-orange-400', productivity: 'text-yellow-400', education: 'text-sky-400',
};

interface PluginItem {
  id: string; name: string; version: string; description: string;
  author: string; enabled: boolean; tools: string[]; skills: string[];
}

const PLUGIN_DATA: PluginItem[] = [
  { id: 'filesystem', name: 'Filesystem Tools', version: '1.0.0', description: 'Read, write, and manage files on the local filesystem.', author: 'Aurelius', enabled: true, tools: ['read_file', 'write_file', 'list_dir', 'search_files'], skills: [] },
  { id: 'web', name: 'Web Tools', version: '1.0.0', description: 'Search the web, fetch URLs, and extract content.', author: 'Aurelius', enabled: true, tools: ['search_web', 'fetch_url', 'extract_content'], skills: [] },
  { id: 'database', name: 'Database Tools', version: '1.0.0', description: 'Query databases, explore schemas, and manage data.', author: 'Aurelius', enabled: true, tools: ['query_db', 'describe_schema', 'list_tables'], skills: [] },
  { id: 'communication', name: 'Communication Tools', version: '1.0.0', description: 'Send emails, messages, and manage notifications.', author: 'Aurelius', enabled: true, tools: ['send_email', 'send_message', 'create_notification'], skills: [] },
  { id: 'system', name: 'System Tools', version: '1.0.0', description: 'Run commands, monitor system, and manage processes.', author: 'Aurelius', enabled: false, tools: ['run_command', 'system_info', 'process_list'], skills: [] },
  { id: 'code', name: 'Code Tools', version: '1.0.0', description: 'Analyze, compile, and run code in multiple languages.', author: 'Aurelius', enabled: true, tools: ['run_code', 'lint_code', 'format_code'], skills: [] },
  { id: 'ai', name: 'AI Tools', version: '1.0.0', description: 'Generate text, embeddings, and interact with LLMs.', author: 'Aurelius', enabled: true, tools: ['generate_text', 'get_embeddings', 'classify_text'], skills: ['prompt_engineering', 'tool_creation'] },
  { id: 'analytics', name: 'Analytics Tools', version: '1.0.0', description: 'Analyze data, create charts, and generate reports.', author: 'Aurelius', enabled: false, tools: ['analyze_data', 'create_chart', 'generate_report'], skills: ['data_analysis', 'data_visualization'] },
  { id: 'security', name: 'Security Tools', version: '1.0.0', description: 'Scan for vulnerabilities, audit logs, check compliance.', author: 'Aurelius', enabled: false, tools: ['security_scan', 'audit_log', 'compliance_check'], skills: ['security_scanning', 'incident_response'] },
  { id: 'devops', name: 'DevOps Tools', version: '1.0.0', description: 'Deploy services, monitor infrastructure, manage containers.', author: 'Aurelius', enabled: false, tools: ['deploy', 'monitor', 'container_exec'], skills: ['deployment', 'infrastructure_monitoring'] },
  { id: 'productivity', name: 'Productivity Tools', version: '1.0.0', description: 'Manage tasks, calendar, notes, and projects.', author: 'Aurelius', enabled: false, tools: ['create_task', 'schedule_event', 'create_note'], skills: ['scheduling', 'task_management', 'note_taking'] },
  { id: 'education', name: 'Education Tools', version: '1.0.0', description: 'Generate quizzes, explain concepts, assess understanding.', author: 'Aurelius', enabled: false, tools: ['generate_quiz', 'explain_concept', 'assess_knowledge'], skills: ['teaching', 'quiz_generation', 'language_learning'] },
];

export default function PluginsPage() {
  const [search, setSearch] = useState('');
  const [selected, setSelected] = useState<PluginItem | null>(null);
  const [plugins, setPlugins] = useState<PluginItem[]>(PLUGIN_DATA);

  const filtered = plugins.filter(p => !search || p.name.toLowerCase().includes(search.toLowerCase()) || p.description.toLowerCase().includes(search.toLowerCase()));

  const togglePlugin = (id: string) => {
    setPlugins(prev => prev.map(p => p.id === id ? { ...p, enabled: !p.enabled } : p));
  };

  const enabled = plugins.filter(p => p.enabled).length;
  const totalTools = plugins.filter(p => p.enabled).reduce((sum, p) => sum + p.tools.length, 0);
  const totalSkills = plugins.filter(p => p.enabled).reduce((sum, p) => sum + p.skills.length, 0);

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
          <Puzzle size={20} className="text-[#4fc3f7]" /> Plugin Manager
          <span className="text-sm font-normal text-[#9e9eb0]">({plugins.length} installed, {enabled} active)</span>
        </h2>
        <div className="relative w-48"><Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#9e9eb0]" /><Input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search plugins..." className="pl-8 py-1.5 text-sm w-full" /></div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-4 gap-3">
        {[
          { label: 'Plugins Installed', value: plugins.length, icon: Puzzle, color: 'text-[#4fc3f7]' },
          { label: 'Active', value: enabled, icon: Power, color: 'text-emerald-400' },
          { label: 'Tools Available', value: totalTools, icon: Wrench, color: 'text-amber-400' },
          { label: 'Skills Provided', value: totalSkills, icon: Star, color: 'text-violet-400' },
        ].map(s => (
          <div key={s.label} className="aurelius-card text-center py-3">
            <s.icon size={16} className={`mx-auto mb-1 ${s.color}`} />
            <p className={`text-lg font-bold ${s.color}`}>{s.value}</p>
            <p className="text-[10px] text-[#9e9eb0] uppercase tracking-wider">{s.label}</p>
          </div>
        ))}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
        {filtered.map(plugin => {
          const Icon = PLUGIN_ICONS[plugin.id] || Puzzle;
          const color = PLUGIN_COLORS[plugin.id] || 'text-[#4fc3f7]';
          return (
            <motion.div key={plugin.id} layout initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              className="aurelius-card p-4 hover:border-[#4fc3f7]/30 transition-all"
            >
              <div className="flex items-start justify-between mb-3">
                <div className="flex items-center gap-3">
                  <div className={`w-10 h-10 rounded-lg flex items-center justify-center ${color} bg-white/5`}>
                    <Icon size={20} />
                  </div>
                  <div>
                    <h3 className="text-sm font-semibold text-[#e0e0e0]">{plugin.name}</h3>
                    <p className="text-[10px] text-[#9e9eb0]">v{plugin.version} by {plugin.author}</p>
                  </div>
                </div>
                <Toggle checked={plugin.enabled} onChange={() => togglePlugin(plugin.id)} />
              </div>
              <p className="text-xs text-[#9e9eb0] mb-3">{plugin.description}</p>
              <div className="grid grid-cols-2 gap-2">
                <div>
                  <p className="text-[9px] text-[#9e9eb0] uppercase tracking-wider mb-1">Tools ({plugin.tools.length})</p>
                  <div className="flex flex-wrap gap-1">
                    {plugin.tools.slice(0, 4).map(t => (
                      <span key={t} className={`text-[9px] px-1.5 py-0.5 rounded ${plugin.enabled ? 'text-emerald-400 bg-emerald-500/10' : 'text-[#9e9eb0] bg-[#0f0f1a]'}`}>{t}</span>
                    ))}
                    {plugin.tools.length > 4 && <span className="text-[9px] text-[#9e9eb0]">+{plugin.tools.length - 4}</span>}
                  </div>
                </div>
                {plugin.skills.length > 0 && (
                  <div>
                    <p className="text-[9px] text-[#9e9eb0] uppercase tracking-wider mb-1">Skills</p>
                    <div className="flex flex-wrap gap-1">
                      {plugin.skills.map(s => (
                        <span key={s} className="text-[9px] text-[#4fc3f7] bg-[#4fc3f7]/10 px-1.5 py-0.5 rounded">{s}</span>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </motion.div>
          );
        })}
      </div>

      {selected && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4" onClick={() => setSelected(null)}>
          <div className="aurelius-card p-6 max-w-md w-full space-y-4" onClick={e => e.stopPropagation()}>
            <h3 className="text-lg font-bold text-[#e0e0e0]">{selected.name}</h3>
            <p className="text-sm text-[#9e9eb0]">{selected.description}</p>
            <button className="aurelius-btn-primary w-full">Configure Plugin</button>
          </div>
        </div>
      )}
    </div>
  );
}
