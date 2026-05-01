import { useState } from 'react';
import { Database, Search, Table, FileJson, Download, Filter } from 'lucide-react';
import Input from '../components/ui/Input';
import { useApi } from '../hooks/useApi';

export default function DataExplorer() {
  const [query, setQuery] = useState('');
  const [collection, setCollection] = useState('agents');
  const { data } = useApi<any[]>(`/data/${collection}${query ? `?q=${encodeURIComponent(query)}` : ''}`);

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2"><Database size={20} className="text-[#4fc3f7]" />Data Explorer</h2>
      <div className="flex gap-2">
        <select value={collection} onChange={e => setCollection(e.target.value)}
          className="bg-[#0f0f1a] border border-[#2d2d44] rounded-lg px-3 py-1.5 text-sm text-[#e0e0e0]">
          <option value="agents">Agents</option><option value="logs">Logs</option><option value="traces">Traces</option>
          <option value="notifications">Notifications</option><option value="workflows">Workflows</option>
        </select>
        <div className="relative flex-1"><Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#9e9eb0]" /><Input value={query} onChange={e => setQuery(e.target.value)} placeholder="Filter..." className="pl-8 py-1.5 text-sm w-full" /></div>
      </div>
      <div className="bg-[#0a0a14] border border-[#2d2d44] rounded-xl overflow-x-auto">
        <pre className="text-xs text-[#9e9eb0] p-4 max-h-[calc(100vh-16rem)] overflow-y-auto">
          {data ? JSON.stringify(data.slice(0, 50), null, 2) : '[]'}
        </pre>
      </div>
    </div>
  );
}
