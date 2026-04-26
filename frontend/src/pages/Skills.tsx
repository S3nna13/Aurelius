import { useState, useCallback } from 'react';
import {
  Wrench,
  Search,
  Play,
  X,
  Code,
  Shield,
  AlertTriangle,
  CheckCircle,
  Clock,
  ChevronRight,
  Loader2,
  Star,
} from 'lucide-react';
import { useApi } from '../hooks/useApi';
import { useToast } from '../components/ToastProvider';
import { useFavorites } from '../hooks/useFavorites';

interface Skill {
  id: string;
  name: string;
  description: string;
  category: string;
  active: boolean;
  version: string | null;
  risk_score: number;
  allow_level: string;
  scope: string;
}

interface SkillDetail extends Skill {
  instructions: string;
  scripts: string[];
  resources: string[];
  metadata: Record<string, unknown>;
}

interface ExecutionResult {
  success: boolean;
  output: string;
  duration_ms: number;
}

export default function Skills() {
  const [search, setSearch] = useState('');
  const [selected, setSelected] = useState<SkillDetail | null>(null);
  const [executing, setExecuting] = useState(false);
  const [execResult, setExecResult] = useState<ExecutionResult | null>(null);
  const [variables, setVariables] = useState<Record<string, string>>({});
  const [showFavoritesOnly, setShowFavoritesOnly] = useState(false);
  const { toast } = useToast();
  const { toggle, isFavorite } = useFavorites();

  const {
    data: skillsData,
    loading,
    error,
    refresh: refreshSkills,
  } = useApi<{ skills: Skill[] }>('/skills', {
    refreshInterval: 10000,
    retries: 2,
    timeout: 8000,
  });

  const skills = skillsData?.skills || [];

  const openDetail = useCallback(async (skillId: string) => {
    setExecResult(null);
    setVariables({});
    try {
      const res = await fetch(`/api/skills/${skillId}`, {
        signal: AbortSignal.timeout(8000),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setSelected(data as SkillDetail);
    } catch (err) {
      toast(err instanceof Error ? err.message : 'Failed to load skill details', 'error');
    }
  }, [toast]);

  const executeSkill = useCallback(async () => {
    if (!selected) return;
    setExecuting(true);
    setExecResult(null);
    try {
      const res = await fetch('/api/skills/execute', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          skill_id: selected.id,
          variables,
        }),
        signal: AbortSignal.timeout(15000),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      const data = await res.json();
      setExecResult(data as ExecutionResult);
      if (data.success) {
        toast('Skill executed successfully', 'success');
      } else {
        toast(data.output || 'Skill execution failed', 'error');
      }
    } catch (err) {
      const msg = err instanceof Error ? err.message : String(err);
      setExecResult({ success: false, output: msg, duration_ms: 0 });
      toast(msg, 'error');
    } finally {
      setExecuting(false);
    }
  }, [selected, variables, toast]);

  const filtered = skills
    .filter(
      (s) =>
        s.name.toLowerCase().includes(search.toLowerCase()) ||
        s.description.toLowerCase().includes(search.toLowerCase()) ||
        s.category.toLowerCase().includes(search.toLowerCase())
    )
    .filter((s) => (showFavoritesOnly ? isFavorite(s.id) : true))
    .sort((a, b) => {
      const aFav = isFavorite(a.id) ? 1 : 0;
      const bFav = isFavorite(b.id) ? 1 : 0;
      return bFav - aFav;
    });

  const riskColor = (score: number) => {
    if (score < 0.15) return 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20';
    if (score < 0.3) return 'text-amber-400 bg-amber-500/10 border-amber-500/20';
    return 'text-rose-400 bg-rose-500/10 border-rose-500/20';
  };

  const categoryIcon = (cat: string) => {
    if (cat === 'Security') return Shield;
    if (cat === 'System') return Code;
    return Wrench;
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
          <Wrench size={20} className="text-[#4fc3f7]" />
          Skills & Plugins
        </h2>
        <div className="flex items-center gap-2">
          <button
            onClick={() => setShowFavoritesOnly((v) => !v)}
            className={`flex items-center gap-1.5 px-3 py-2 rounded-lg text-xs font-medium border transition-colors ${
              showFavoritesOnly
                ? 'bg-amber-500/10 text-amber-400 border-amber-500/30'
                : 'bg-[#0f0f1a] text-[#9e9eb0] border-[#2d2d44] hover:border-aurelius-accent/30'
            }`}
          >
            <Star size={14} className={showFavoritesOnly ? 'fill-amber-400' : ''} />
            Favorites
          </button>
          <div className="relative">
            <Search
              size={14}
              className="absolute left-3 top-1/2 -translate-y-1/2 text-[#9e9eb0]"
            />
            <input
              type="text"
              placeholder="Search skills..."
              value={search}
              onChange={(e) => setSearch(e.target.value)}
              className="bg-[#0f0f1a] border border-[#2d2d44] rounded-lg pl-9 pr-4 py-2 text-sm text-[#e0e0e0] placeholder:text-[#9e9eb0] focus:outline-none focus:border-[#4fc3f7] w-full sm:w-64"
            />
          </div>
        </div>
      </div>

      {loading && (
        <div className="aurelius-card text-center py-12 text-[#9e9eb0]">
          <Loader2 size={32} className="mx-auto mb-3 animate-spin opacity-60" />
          <p>Loading skills...</p>
        </div>
      )}

      {error && !loading && (
        <div className="aurelius-card border-rose-500/30 bg-rose-500/5 text-rose-300">
          <AlertTriangle size={18} className="inline mr-2" />
          {error.message}
          <button
            onClick={refreshSkills}
            className="ml-4 text-xs underline hover:text-rose-200"
          >
            Retry
          </button>
        </div>
      )}

      {!loading && (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filtered.map((skill) => {
            const Icon = categoryIcon(skill.category);
            return (
              <div
                key={skill.id}
                onClick={() => openDetail(skill.id)}
                className="aurelius-card flex flex-col gap-3 hover:border-[#4fc3f7]/30 transition-colors cursor-pointer group"
              >
                <div className="flex items-start justify-between">
                  <div className="flex items-center gap-3">
                    <div className="w-9 h-9 rounded-lg bg-[#4fc3f7]/10 text-[#4fc3f7] flex items-center justify-center border border-[#4fc3f7]/20">
                      <Icon size={18} />
                    </div>
                    <div>
                      <h3 className="text-sm font-semibold text-[#e0e0e0] group-hover:text-[#4fc3f7] transition-colors">
                        {skill.name}
                      </h3>
                      <p className="text-[10px] text-[#9e9eb0] uppercase tracking-wider">
                        {skill.category} · {skill.scope}
                      </p>
                    </div>
                  </div>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        toggle(skill.id);
                      }}
                      className={`p-1.5 rounded-lg transition-colors ${
                        isFavorite(skill.id)
                          ? 'text-amber-400 hover:bg-amber-500/10'
                          : 'text-[#9e9eb0] hover:text-amber-400 hover:bg-aurelius-border/40'
                      }`}
                      title={isFavorite(skill.id) ? 'Remove from favorites' : 'Add to favorites'}
                    >
                      <Star size={14} className={isFavorite(skill.id) ? 'fill-amber-400' : ''} />
                    </button>
                    <ChevronRight
                      size={16}
                      className="text-[#9e9eb0] group-hover:text-[#4fc3f7] transition-colors"
                    />
                  </div>
                </div>
                <p className="text-sm text-[#9e9eb0] leading-relaxed">
                  {skill.description}
                </p>
                <div className="flex items-center gap-2 mt-auto flex-wrap">
                  <span
                    className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${
                      skill.active
                        ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                        : 'bg-[#2d2d44]/20 text-[#9e9eb0] border-[#2d2d44]/40'
                    }`}
                  >
                    {skill.active ? 'Enabled' : 'Disabled'}
                  </span>
                  <span
                    className={`text-[10px] font-bold px-2 py-0.5 rounded-full border ${riskColor(
                      skill.risk_score
                    )}`}
                  >
                    Risk {(skill.risk_score * 100).toFixed(0)}%
                  </span>
                  {skill.version && (
                    <span className="text-[10px] text-[#9e9eb0] bg-[#2d2d44]/30 px-2 py-0.5 rounded-full border border-[#2d2d44]/40">
                      v{skill.version}
                    </span>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}

      {filtered.length === 0 && !loading && (
        <div className="aurelius-card text-center py-12 text-[#9e9eb0]">
          <Search size={32} className="mx-auto mb-3 opacity-40" />
          <p>No skills match your search.</p>
        </div>
      )}

      {/* Detail Modal */}
      {selected && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/60 backdrop-blur-sm">
          <div className="bg-[#1a1a2e] border border-[#2d2d44] rounded-xl w-full max-w-2xl max-h-[90vh] overflow-y-auto shadow-2xl">
            <div className="flex items-center justify-between p-5 border-b border-[#2d2d44]">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-lg bg-[#4fc3f7]/10 text-[#4fc3f7] flex items-center justify-center border border-[#4fc3f7]/20">
                  <Wrench size={20} />
                </div>
                <div>
                  <h3 className="text-base font-bold text-[#e0e0e0]">
                    {selected.name}
                  </h3>
                  <p className="text-xs text-[#9e9eb0]">
                    {selected.category} · {selected.scope}
                  </p>
                </div>
              </div>
              <button
                onClick={() => setSelected(null)}
                className="text-[#9e9eb0] hover:text-[#e0e0e0] transition-colors"
              >
                <X size={20} />
              </button>
            </div>

            <div className="p-5 space-y-5">
              <p className="text-sm text-[#9e9eb0]">{selected.description}</p>

              <div className="flex flex-wrap gap-2">
                <span
                  className={`text-xs font-bold px-2.5 py-1 rounded-full border ${
                    selected.active
                      ? 'bg-emerald-500/10 text-emerald-400 border-emerald-500/20'
                      : 'bg-[#2d2d44]/20 text-[#9e9eb0] border-[#2d2d44]/40'
                  }`}
                >
                  {selected.active ? 'Enabled' : 'Disabled'}
                </span>
                <span
                  className={`text-xs font-bold px-2.5 py-1 rounded-full border ${riskColor(
                    selected.risk_score
                  )}`}
                >
                  Risk {(selected.risk_score * 100).toFixed(0)}%
                </span>
                <span className="text-xs text-[#9e9eb0] bg-[#2d2d44]/30 px-2.5 py-1 rounded-full border border-[#2d2d44]/40">
                  {selected.allow_level}
                </span>
                {selected.version && (
                  <span className="text-xs text-[#9e9eb0] bg-[#2d2d44]/30 px-2.5 py-1 rounded-full border border-[#2d2d44]/40">
                    v{selected.version}
                  </span>
                )}
              </div>

              {selected.instructions && (
                <div>
                  <h4 className="text-xs font-bold text-[#9e9eb0] uppercase tracking-wider mb-2">
                    Instructions
                  </h4>
                  <div className="bg-[#0f0f1a] border border-[#2d2d44] rounded-lg p-3 text-sm text-[#e0e0e0] font-mono whitespace-pre-wrap max-h-48 overflow-y-auto">
                    {selected.instructions}
                  </div>
                </div>
              )}

              {selected.scripts.length > 0 && (
                <div>
                  <h4 className="text-xs font-bold text-[#9e9eb0] uppercase tracking-wider mb-2">
                    Scripts
                  </h4>
                  <ul className="space-y-1">
                    {selected.scripts.map((s) => (
                      <li
                        key={s}
                        className="text-sm text-[#e0e0e0] flex items-center gap-2"
                      >
                        <Code size={14} className="text-[#4fc3f7]" />
                        {s}
                      </li>
                    ))}
                  </ul>
                </div>
              )}

              {/* Variable Input */}
              <div>
                <h4 className="text-xs font-bold text-[#9e9eb0] uppercase tracking-wider mb-2">
                  Variables
                </h4>
                <div className="flex gap-2">
                  <input
                    type="text"
                    placeholder="key"
                    value={Object.keys(variables).pop() || ''}
                    onChange={(e) => {
                      const key = e.target.value;
                      const val = Object.values(variables).pop() || '';
                      const rest = Object.fromEntries(
                        Object.entries(variables).slice(0, -1)
                      );
                      setVariables({ ...rest, [key]: val });
                    }}
                    className="flex-1 bg-[#0f0f1a] border border-[#2d2d44] rounded-lg px-3 py-2 text-sm text-[#e0e0e0] placeholder:text-[#9e9eb0] focus:outline-none focus:border-[#4fc3f7]"
                  />
                  <input
                    type="text"
                    placeholder="value"
                    value={Object.values(variables).pop() || ''}
                    onChange={(e) => {
                      const keys = Object.keys(variables);
                      const key = keys.pop() || 'var';
                      const rest = Object.fromEntries(
                        Object.entries(variables).slice(0, -1)
                      );
                      setVariables({ ...rest, [key]: e.target.value });
                    }}
                    className="flex-1 bg-[#0f0f1a] border border-[#2d2d44] rounded-lg px-3 py-2 text-sm text-[#e0e0e0] placeholder:text-[#9e9eb0] focus:outline-none focus:border-[#4fc3f7]"
                  />
                </div>
                {Object.entries(variables).length > 0 &&
                  Object.entries(variables).some(([k]) => k) && (
                    <div className="mt-2 flex flex-wrap gap-2">
                      {Object.entries(variables)
                        .filter(([k]) => k)
                        .map(([k, v]) => (
                          <span
                            key={k}
                            className="text-xs bg-[#4fc3f7]/10 text-[#4fc3f7] border border-[#4fc3f7]/20 px-2 py-1 rounded-full"
                          >
                            {k}={String(v)}
                          </span>
                        ))}
                    </div>
                  )}
              </div>

              {/* Execute Button */}
              <button
                onClick={executeSkill}
                disabled={executing}
                className="aurelius-btn w-full flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {executing ? (
                  <Loader2 size={16} className="animate-spin" />
                ) : (
                  <Play size={16} />
                )}
                {executing ? 'Executing...' : 'Execute Skill'}
              </button>

              {/* Execution Result */}
              {execResult && (
                <div
                  className={`border rounded-lg p-4 ${
                    execResult.success
                      ? 'bg-emerald-500/5 border-emerald-500/20'
                      : 'bg-rose-500/5 border-rose-500/20'
                  }`}
                >
                  <div className="flex items-center gap-2 mb-2">
                    {execResult.success ? (
                      <CheckCircle size={16} className="text-emerald-400" />
                    ) : (
                      <AlertTriangle size={16} className="text-rose-400" />
                    )}
                    <span
                      className={`text-sm font-bold ${
                        execResult.success ? 'text-emerald-400' : 'text-rose-400'
                      }`}
                    >
                      {execResult.success ? 'Success' : 'Failed'}
                    </span>
                    <span className="text-xs text-[#9e9eb0] flex items-center gap-1 ml-auto">
                      <Clock size={12} />
                      {execResult.duration_ms.toFixed(1)} ms
                    </span>
                  </div>
                  <pre className="text-xs text-[#e0e0e0] font-mono whitespace-pre-wrap bg-[#0f0f1a] rounded-lg p-3 border border-[#2d2d44]">
                    {execResult.output}
                  </pre>
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
