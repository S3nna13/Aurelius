import { useState } from 'react';
import { BookOpen, Search, ChevronDown, ChevronRight, Copy, CheckCircle, Code } from 'lucide-react';
import Input from '../components/ui/Input';
import { useApi } from '../hooks/useApi';
import { useToast } from '../components/ToastProvider';

interface Endpoint { path: string; method: string; summary: string; parameters: any[]; }

export default function ApiDocs() {
  const { toast } = useToast();
  const [search, setSearch] = useState('');
  const [expanded, setExpanded] = useState<string | null>(null);
  const { data } = useApi<{ endpoints: Endpoint[] }>('/openapi.json');
  const endpoints = (data?.endpoints || []).filter(e => !search || e.path.includes(search) || e.summary?.toLowerCase().includes(search.toLowerCase()));

  const METHOD_COLORS: Record<string, string> = { GET: 'text-emerald-400', POST: 'text-[#4fc3f7]', PUT: 'text-amber-400', DELETE: 'text-rose-400', PATCH: 'text-violet-400' };

  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2"><BookOpen size={20} className="text-[#4fc3f7]" />API Documentation</h2>
        <div className="relative w-64"><Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#9e9eb0]" /><Input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search endpoints..." className="pl-8 py-1.5 text-sm" /></div>
      </div>
      <div className="space-y-1">
        {endpoints.map(ep => {
          const isExpanded = expanded === ep.path + ep.method;
          return (
            <div key={ep.path + ep.method} className="aurelius-card overflow-hidden">
              <div className="flex items-center gap-3 p-3 cursor-pointer hover:bg-white/[0.02]" onClick={() => setExpanded(isExpanded ? null : ep.path + ep.method)}>
                {isExpanded ? <ChevronDown size={14} className="text-[#9e9eb0]" /> : <ChevronRight size={14} className="text-[#9e9eb0]" />}
                <span className={`text-[10px] font-bold w-14 ${METHOD_COLORS[ep.method] || 'text-gray-400'}`}>{ep.method}</span>
                <span className="text-sm font-mono text-[#e0e0e0]">{ep.path}</span>
                {ep.summary && <span className="text-xs text-[#9e9eb0] ml-2 truncate">{ep.summary}</span>}
              </div>
              {isExpanded && (
                <div className="px-3 pb-3 border-t border-[#2d2d44] pt-3">
                  <div className="flex gap-2 mb-2">
                    <button onClick={() => { navigator.clipboard.writeText(`${ep.method} ${ep.path}`); toast('Copied', 'success'); }}
                      className="text-[10px] text-[#4fc3f7] bg-[#4fc3f7]/10 px-2 py-1 rounded flex items-center gap-1"><Copy size={10} /> Copy</button>
                  </div>
                  {ep.parameters?.length > 0 && (
                    <div><p className="text-xs font-bold text-[#e0e0e0] mb-1">Parameters</p>
                      {ep.parameters.map((p: any, i: number) => (
                        <div key={i} className="text-xs text-[#9e9eb0] font-mono bg-[#0a0a14] p-1.5 rounded mb-1">{p.name}: {p.type || 'string'}{p.required ? ' (required)' : ''}</div>
                      ))}
                    </div>
                  )}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
