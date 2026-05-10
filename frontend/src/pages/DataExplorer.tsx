import { useState } from 'react';
import { Database, Search } from 'lucide-react';
import Input from '../components/ui/Input';
import { useApi } from '../hooks/useApi';
import { DataGrid, type DataGridColumn } from '../components/DataGrid';

type Category = 'agents' | 'logs' | 'notifications' | 'traces';

interface Agent {
  id: string;
  name: string;
  role: string;
  state: string;
  created: number;
}

interface LogEntry {
  timestamp: string;
  level: string;
  module: string;
  message: string;
}

interface Notification {
  id: string;
  title: string;
  body: string;
  priority: string;
  read: boolean;
}

interface Trace {
  id: string;
  type: string;
  status: string;
  timestamp: number;
}

const CATEGORY_COLUMNS: Record<Category, DataGridColumn<any>[]> = {
  agents: [
    { key: 'id', header: 'ID', width: '200px', render: (a: Agent) => <span className="font-mono text-xs text-[#4fc3f7]">{a.id}</span> },
    { key: 'name', header: 'Name', sortable: true },
    { key: 'role', header: 'Role', sortable: true },
    { key: 'state', header: 'State', render: (a: Agent) => <span className={`text-xs px-2 py-0.5 rounded-full ${a.state === 'active' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-[#0f0f1a] text-[#9e9eb0]'}`}>{a.state}</span> },
    { key: 'created', header: 'Created', render: (a: Agent) => <span className="text-[#9e9eb0]">{new Date(a.created).toLocaleString()}</span> },
  ],
  logs: [
    { key: 'timestamp', header: 'Timestamp', width: '160px', render: (l: LogEntry) => <span className="text-[#9e9eb0] font-mono text-xs">{l.timestamp}</span> },
    { key: 'level', header: 'Level', width: '70px', render: (l: LogEntry) => <span className={`font-bold text-xs ${l.level === 'ERROR' ? 'text-rose-400' : l.level === 'WARN' ? 'text-amber-400' : 'text-[#4fc3f7]'}`}>{l.level}</span> },
    { key: 'module', header: 'Module', width: '120px', render: (l: LogEntry) => <span className="text-[#4fc3f7]/60">{l.module}</span> },
    { key: 'message', header: 'Message' },
  ],
  notifications: [
    { key: 'id', header: 'ID', width: '200px', render: (n: Notification) => <span className="font-mono text-xs text-[#4fc3f7]">{n.id}</span> },
    { key: 'title', header: 'Title', sortable: true },
    { key: 'priority', header: 'Priority', width: '80px', render: (n: Notification) => <span className={`text-xs px-2 py-0.5 rounded-full ${n.priority === 'high' ? 'bg-rose-500/10 text-rose-400' : n.priority === 'medium' ? 'bg-amber-500/10 text-amber-400' : 'bg-[#0f0f1a] text-[#9e9eb0]'}`}>{n.priority}</span> },
    { key: 'read', header: 'Read', width: '60px', render: (n: Notification) => <span className={n.read ? 'text-emerald-400' : 'text-[#9e9eb0]'}>{(n.read ? 'Yes' : 'No')}</span> },
    { key: 'body', header: 'Body' },
  ],
  traces: [
    { key: 'id', header: 'ID', width: '200px', render: (t: Trace) => <span className="font-mono text-xs text-[#4fc3f7]">{t.id}</span> },
    { key: 'type', header: 'Type', sortable: true },
    { key: 'status', header: 'Status', render: (t: Trace) => <span className={`text-xs px-2 py-0.5 rounded-full ${t.status === 'success' ? 'bg-emerald-500/10 text-emerald-400' : 'bg-rose-500/10 text-rose-400'}`}>{t.status}</span> },
    { key: 'timestamp', header: 'Timestamp', render: (t: Trace) => <span className="text-[#9e9eb0]">{new Date(t.timestamp).toLocaleString()}</span> },
  ],
};

const API_PATHS: Record<Category, string> = {
  agents: '/api/agents',
  logs: '/api/logs',
  notifications: '/api/notifications',
  traces: '/api/traces',
};

export default function DataExplorer() {
  const [category, setCategory] = useState<Category>('agents');
  const [search, setSearch] = useState('');
  const { data, loading } = useApi<any[]>(API_PATHS[category]);

  return (
    <div className="space-y-4">
      <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2"><Database size={20} className="text-[#4fc3f7]" />Data Explorer</h2>
      <div className="flex gap-2">
        <select
          value={category}
          onChange={e => setCategory(e.target.value as Category)}
          className="bg-[#0f0f1a] border border-[#2d2d44] rounded-lg px-3 py-1.5 text-sm text-[#e0e0e0]"
        >
          <option value="agents">Agents</option>
          <option value="logs">Logs</option>
          <option value="traces">Traces</option>
          <option value="notifications">Notifications</option>
        </select>
        <div className="relative flex-1">
          <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#9e9eb0]" />
          <Input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="Filter..."
            className="pl-8 py-1.5 text-sm w-full"
          />
        </div>
      </div>
      <DataGrid
        columns={CATEGORY_COLUMNS[category]}
        data={(data || []).slice(0, 100)}
        keyField="id"
        loading={loading}
        searchable
        searchPlaceholder="Filter data..."
        emptyMessage="No data available."
      />
    </div>
  );
}
