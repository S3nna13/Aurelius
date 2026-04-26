import { useState } from 'react';
import { Wrench, Search, ToggleLeft, ToggleRight, Puzzle, Globe, Database, Lock } from 'lucide-react';

interface Skill {
  id: number;
  name: string;
  description: string;
  icon: typeof Wrench;
  enabled: boolean;
  category: string;
}

const initialSkills: Skill[] = [
  { id: 1, name: 'Web Scraping', description: 'Extract structured data from web pages.', icon: Globe, enabled: true, category: 'Data' },
  { id: 2, name: 'Database Query', description: 'Run SQL and NoSQL queries safely.', icon: Database, enabled: true, category: 'Data' },
  { id: 3, name: 'File System', description: 'Read, write, and manage local files.', icon: Puzzle, enabled: true, category: 'System' },
  { id: 4, name: 'Auth Manager', description: 'Handle tokens, SSO, and credentials.', icon: Lock, enabled: false, category: 'Security' },
  { id: 5, name: 'API Client', description: 'Make HTTP requests to external APIs.', icon: Globe, enabled: true, category: 'Network' },
  { id: 6, name: 'Memory Search', description: 'Semantic search across memory layers.', icon: Database, enabled: true, category: 'AI' },
];

export default function Skills() {
  const [skills, setSkills] = useState<Skill[]>(initialSkills);
  const [search, setSearch] = useState('');

  const toggle = (id: number) => {
    setSkills((prev) =>
      prev.map((s) => (s.id === id ? { ...s, enabled: !s.enabled } : s))
    );
  };

  const filtered = skills.filter(
    (s) =>
      s.name.toLowerCase().includes(search.toLowerCase()) ||
      s.description.toLowerCase().includes(search.toLowerCase())
  );

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <h2 className="text-lg font-bold text-aurelius-text flex items-center gap-2">
          <Wrench size={20} className="text-aurelius-accent" />
          Skills & Plugins
        </h2>
        <div className="relative">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-aurelius-muted" />
          <input
            type="text"
            placeholder="Search skills..."
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            className="bg-aurelius-bg border border-aurelius-border rounded-lg pl-9 pr-4 py-2 text-sm text-aurelius-text placeholder:text-aurelius-muted focus:outline-none focus:border-aurelius-accent w-full sm:w-64"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {filtered.map((skill) => (
          <div
            key={skill.id}
            className="aurelius-card flex flex-col gap-3 hover:border-aurelius-accent/30 transition-colors"
          >
            <div className="flex items-start justify-between">
              <div className="flex items-center gap-3">
                <div className="w-9 h-9 rounded-lg bg-aurelius-accent/10 text-aurelius-accent flex items-center justify-center border border-aurelius-accent/20">
                  <skill.icon size={18} />
                </div>
                <div>
                  <h3 className="text-sm font-semibold text-aurelius-text">{skill.name}</h3>
                  <p className="text-[10px] text-aurelius-muted uppercase tracking-wider">{skill.category}</p>
                </div>
              </div>
              <button
                onClick={() => toggle(skill.id)}
                className={`transition-colors ${skill.enabled ? 'text-aurelius-accent' : 'text-aurelius-muted'}`}
              >
                {skill.enabled ? <ToggleRight size={24} /> : <ToggleLeft size={24} />}
              </button>
            </div>
            <p className="text-sm text-aurelius-muted leading-relaxed">{skill.description}</p>
            <div className="flex items-center gap-2 mt-auto">
              <span
                className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${
                  skill.enabled
                    ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                    : 'bg-aurelius-border/20 text-aurelius-muted border-aurelius-border/40'
                }`}
              >
                {skill.enabled ? 'Enabled' : 'Disabled'}
              </span>
            </div>
          </div>
        ))}
      </div>

      {filtered.length === 0 && (
        <div className="aurelius-card text-center py-12 text-aurelius-muted">
          <Search size={32} className="mx-auto mb-3 opacity-40" />
          <p>No skills match your search.</p>
        </div>
      )}
    </div>
  );
}
