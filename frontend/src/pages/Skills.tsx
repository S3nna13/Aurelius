import { useState } from 'react';
import { motion } from 'framer-motion';
import { Puzzle, Search, Filter, Code, BookOpen, Server, MessageSquare, Pen, GraduationCap, Database, Calendar, Wrench, Star } from 'lucide-react';
import Input from '../components/ui/Input';
import EmptyState from '../components/EmptyState';
import Skeleton from '../components/Skeleton';

const CATEGORY_ICONS: Record<string, typeof Puzzle> = {
  coding: Code, research: BookOpen, devops: Server, communication: MessageSquare,
  creative: Pen, education: GraduationCap, data: Database, productivity: Calendar, meta: Wrench,
};
const CATEGORY_COLORS: Record<string, string> = {
  coding: 'text-emerald-400', research: 'text-[#4fc3f7]', devops: 'text-amber-400',
  communication: 'text-violet-400', creative: 'text-rose-400', education: 'text-cyan-400',
  data: 'text-indigo-400', productivity: 'text-lime-400', meta: 'text-gray-400',
};

const ALL_CATEGORIES = ['coding', 'research', 'devops', 'communication', 'creative', 'education', 'data', 'productivity', 'meta'];

interface SkillItem { id: string; name: string; category: string; description: string; tags: string[]; agent_types?: string[]; }

const SKILL_DATA: SkillItem[] = [
  { id: 'code_generation', name: 'Code Generation', category: 'coding', description: 'Generate code in any language from natural language descriptions.', tags: ['python', 'typescript', 'rust'] },
  { id: 'code_review', name: 'Code Review', category: 'coding', description: 'Review PRs for bugs, style, and security vulnerabilities.', tags: ['quality', 'security'] },
  { id: 'debugging', name: 'Debugging', category: 'coding', description: 'Find and fix bugs with systematic root cause analysis.', tags: ['testing', 'quality'] },
  { id: 'refactoring', name: 'Refactoring', category: 'coding', description: 'Restructure code for better maintainability without changing behavior.', tags: ['quality', 'patterns'] },
  { id: 'unit_testing', name: 'Unit Testing', category: 'coding', description: 'Generate and run unit, integration, and e2e tests.', tags: ['testing', 'coverage'] },
  { id: 'static_analysis', name: 'Static Analysis', category: 'coding', description: 'Analyze code without running it to find potential issues.', tags: ['quality', 'security'] },
  { id: 'web_search', name: 'Web Search', category: 'research', description: 'Search the web and synthesize findings into coherent results.', tags: ['search', 'information'] },
  { id: 'fact_checking', name: 'Fact Checking', category: 'research', description: 'Verify claims against authoritative sources with confidence scoring.', tags: ['accuracy', 'verification'] },
  { id: 'data_analysis', name: 'Data Analysis', category: 'research', description: 'Analyze datasets and extract actionable insights.', tags: ['statistics', 'analytics'] },
  { id: 'summarization', name: 'Summarization', category: 'research', description: 'Condense long texts into concise, accurate summaries.', tags: ['reading', 'efficiency'] },
  { id: 'infrastructure_monitoring', name: 'Infrastructure Monitoring', category: 'devops', description: 'Monitor system health, track metrics, and alert on anomalies.', tags: ['observability', 'alerts'] },
  { id: 'deployment', name: 'Deployment', category: 'devops', description: 'Deploy services, manage releases, and rollback when needed.', tags: ['ci/cd', 'kubernetes'] },
  { id: 'incident_response', name: 'Incident Response', category: 'devops', description: 'Respond to and resolve system incidents with runbooks.', tags: ['reliability', 'sre'] },
  { id: 'security_scanning', name: 'Security Scanning', category: 'devops', description: 'Scan for vulnerabilities, misconfigurations, and compliance gaps.', tags: ['vulnerability', 'compliance'] },
  { id: 'log_analysis', name: 'Log Analysis', category: 'devops', description: 'Parse and analyze log files for patterns and anomalies.', tags: ['observability', 'debugging'] },
  { id: 'email_drafting', name: 'Email Drafting', category: 'communication', description: 'Draft professional emails for any context or audience.', tags: ['writing', 'professional'] },
  { id: 'content_creation', name: 'Content Creation', category: 'communication', description: 'Create engaging content for any platform or medium.', tags: ['marketing', 'writing'] },
  { id: 'customer_support', name: 'Customer Support', category: 'communication', description: 'Handle customer inquiries, triage issues, and provide solutions.', tags: ['service', 'ticketing'] },
  { id: 'translation', name: 'Translation', category: 'communication', description: 'Translate text between languages while preserving meaning.', tags: ['languages', 'international'] },
  { id: 'creative_writing', name: 'Creative Writing', category: 'creative', description: 'Write stories, poems, and creative content with vivid language.', tags: ['writing', 'storytelling'] },
  { id: 'copywriting', name: 'Copywriting', category: 'creative', description: 'Write persuasive marketing copy and advertising content.', tags: ['marketing', 'writing'] },
  { id: 'brainstorming', name: 'Brainstorming', category: 'creative', description: 'Generate creative ideas and novel solutions to problems.', tags: ['ideation', 'creativity'] },
  { id: 'teaching', name: 'Teaching', category: 'education', description: 'Explain concepts with adaptive pedagogy and Socratic dialogue.', tags: ['learning', 'pedagogy'] },
  { id: 'quiz_generation', name: 'Quiz Generation', category: 'education', description: 'Create quizzes and assessments to test knowledge.', tags: ['assessment', 'learning'] },
  { id: 'language_learning', name: 'Language Learning', category: 'education', description: 'Practice conversations, grammar, and vocabulary in new languages.', tags: ['languages', 'practice'] },
  { id: 'sql_querying', name: 'SQL Querying', category: 'data', description: 'Write, optimize, and debug SQL queries across databases.', tags: ['database', 'analytics'] },
  { id: 'data_visualization', name: 'Data Visualization', category: 'data', description: 'Create charts, graphs, and dashboards from any data.', tags: ['charts', 'dashboard'] },
  { id: 'database_design', name: 'Database Design', category: 'data', description: 'Design schemas, optimize queries, and plan migrations.', tags: ['schema', 'optimization'] },
  { id: 'scheduling', name: 'Scheduling', category: 'productivity', description: 'Manage calendars, schedule events, and resolve conflicts.', tags: ['calendar', 'time'] },
  { id: 'task_management', name: 'Task Management', category: 'productivity', description: 'Track tasks, manage sprints, and organize workflows.', tags: ['agile', 'organization'] },
  { id: 'note_taking', name: 'Note Taking', category: 'productivity', description: 'Take, organize, and summarize meeting notes and documents.', tags: ['documentation', 'organization'] },
  { id: 'prompt_engineering', name: 'Prompt Engineering', category: 'meta', description: 'Design and optimize prompts for LLMs across use cases.', tags: ['optimization', 'llm'] },
  { id: 'tool_creation', name: 'Tool Creation', category: 'meta', description: 'Create new tools and integrations for the agent platform.', tags: ['development', 'extensibility'] },
  { id: 'workflow_automation', name: 'Workflow Automation', category: 'meta', description: 'Automate multi-step workflows across tools and services.', tags: ['automation', 'efficiency'] },
];

export default function SkillsRegistry() {
  const [search, setSearch] = useState('');
  const [category, setCategory] = useState('all');
  const [selected, setSelected] = useState<SkillItem | null>(null);

  const filtered = SKILL_DATA.filter(s => {
    if (category !== 'all' && s.category !== category) return false;
    return !search || s.name.toLowerCase().includes(search.toLowerCase()) || s.description.toLowerCase().includes(search.toLowerCase()) || s.tags.some(t => t.includes(search.toLowerCase()));
  });

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
          <Puzzle size={20} className="text-[#4fc3f7]" /> Skills Registry
          <span className="text-sm font-normal text-[#9e9eb0]">({SKILL_DATA.length} total)</span>
        </h2>
        <div className="flex gap-2">
          <div className="relative"><Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#9e9eb0]" /><Input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search skills..." className="pl-8 py-1.5 text-sm w-48" /></div>
          <select value={category} onChange={e => setCategory(e.target.value)} className="bg-[#0f0f1a] border border-[#2d2d44] rounded-lg px-3 py-1.5 text-sm text-[#e0e0e0]">
            <option value="all">All Categories</option>
            {ALL_CATEGORIES.map(c => <option key={c} value={c}>{c.charAt(0).toUpperCase() + c.slice(1)}</option>)}
          </select>
        </div>
      </div>

      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-9 gap-2">
        {ALL_CATEGORIES.map(c => {
          const Icon = CATEGORY_ICONS[c] || Puzzle;
          const count = SKILL_DATA.filter(s => s.category === c).length;
          return (
            <button key={c} onClick={() => setCategory(c)}
              className={`aurelius-card p-2 text-center hover:border-[#4fc3f7]/30 transition-all ${category === c ? 'border-[#4fc3f7]/50' : ''}`}>
              <Icon size={14} className={`mx-auto mb-0.5 ${CATEGORY_COLORS[c] || 'text-[#4fc3f7]'}`} />
              <p className="text-[9px] text-[#e0e0e0] font-medium truncate">{c}</p>
              <p className="text-[8px] text-[#9e9eb0]">{count} skills</p>
            </button>
          );
        })}
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-3">
        {filtered.map(skill => {
          const Icon = CATEGORY_ICONS[skill.category] || Puzzle;
          const color = CATEGORY_COLORS[skill.category] || 'text-[#4fc3f7]';
          return (
            <motion.div key={skill.id} layout initial={{ opacity: 0 }} animate={{ opacity: 1 }}
              onClick={() => setSelected(skill)}
              className="aurelius-card p-4 hover:border-[#4fc3f7]/30 cursor-pointer transition-all group"
            >
              <div className="flex items-start gap-3 mb-2">
                <div className={`w-8 h-8 rounded-lg flex items-center justify-center ${color} bg-white/5`}><Icon size={14} /></div>
                <div className="min-w-0 flex-1">
                  <h3 className="text-sm font-semibold text-[#e0e0e0] truncate">{skill.name}</h3>
                  <span className={`text-[9px] ${color}`}>{skill.category}</span>
                </div>
              </div>
              <p className="text-xs text-[#9e9eb0] line-clamp-2 mb-2">{skill.description}</p>
              <div className="flex flex-wrap gap-1">
                {skill.tags.slice(0, 4).map(tag => (
                  <span key={tag} className="text-[9px] text-[#4fc3f7] bg-[#4fc3f7]/10 px-1.5 py-0.5 rounded">{tag}</span>
                ))}
              </div>
            </motion.div>
          );
        })}
      </div>

      {selected && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4" onClick={() => setSelected(null)}>
          <div className="aurelius-card p-6 max-w-md w-full space-y-4" onClick={e => e.stopPropagation()}>
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-xl bg-[#4fc3f7]/10 flex items-center justify-center"><Puzzle size={20} className="text-[#4fc3f7]" /></div>
              <div>
                <h3 className="text-lg font-bold text-[#e0e0e0]">{selected.name}</h3>
                <span className={`text-xs ${CATEGORY_COLORS[selected.category] || 'text-[#4fc3f7]'}`}>{selected.category}</span>
              </div>
            </div>
            <p className="text-sm text-[#9e9eb0]">{selected.description}</p>
            <div className="flex flex-wrap gap-1.5">
              {selected.tags.map(tag => <span key={tag} className="text-[10px] text-[#4fc3f7] bg-[#4fc3f7]/10 px-2 py-0.5 rounded-full">{tag}</span>)}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
