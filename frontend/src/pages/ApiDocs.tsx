import { useState } from 'react';
import { BookOpen, Copy, CheckCircle2, Server, ChevronDown, ChevronRight } from 'lucide-react';

interface Endpoint {
  method: string;
  path: string;
  description: string;
  params?: { name: string; type: string; required: boolean; description: string }[];
  response?: string;
}

const endpoints: { category: string; items: Endpoint[] }[] = [
  {
    category: 'System',
    items: [
      { method: 'GET', path: '/api/health', description: 'Health check endpoint' },
      { method: 'GET', path: '/api/status', description: 'System status with agents, skills, plugins' },
      { method: 'GET', path: '/api/activity', description: 'Activity log entries' },
      { method: 'GET', path: '/api/logs', description: 'System logs with filtering', params: [{ name: 'level', type: 'string', required: false, description: 'Filter by log level' }, { name: 'q', type: 'string', required: false, description: 'Search query' }, { name: 'limit', type: 'number', required: false, description: 'Max entries (default 100)' }] },
    ],
  },
  {
    category: 'Notifications',
    items: [
      { method: 'GET', path: '/api/notifications', description: 'List all notifications' },
      { method: 'GET', path: '/api/notifications/stats', description: 'Notification statistics' },
      { method: 'POST', path: '/api/notifications/read', description: 'Mark a notification as read', params: [{ name: 'id', type: 'string', required: true, description: 'Notification ID' }] },
      { method: 'POST', path: '/api/notifications/read-all', description: 'Mark all notifications as read' },
      { method: 'GET', path: '/api/notifications/preferences', description: 'Get notification preferences' },
      { method: 'POST', path: '/api/notifications/preferences', description: 'Update notification preferences', params: [{ name: 'preferences', type: 'object', required: true, description: 'Channel preferences object' }] },
    ],
  },
  {
    category: 'Skills',
    items: [
      { method: 'GET', path: '/api/skills', description: 'List all skills' },
      { method: 'GET', path: '/api/skills/:id', description: 'Get skill details' },
      { method: 'POST', path: '/api/skills/execute', description: 'Execute a skill', params: [{ name: 'skill_id', type: 'string', required: true, description: 'Skill ID' }, { name: 'variables', type: 'object', required: false, description: 'Execution variables' }] },
    ],
  },
  {
    category: 'Workflows',
    items: [
      { method: 'GET', path: '/api/workflows', description: 'List all workflows' },
      { method: 'GET', path: '/api/workflows/:id', description: 'Get workflow details' },
      { method: 'POST', path: '/api/workflows/:id/trigger', description: 'Trigger a workflow action', params: [{ name: 'action', type: 'string', required: true, description: 'Action to trigger' }] },
    ],
  },
  {
    category: 'Memory',
    items: [
      { method: 'GET', path: '/api/memory', description: 'Memory layer summary' },
      { method: 'GET', path: '/api/memory/entries', description: 'Memory entries with filtering', params: [{ name: 'layer', type: 'string', required: false, description: 'Layer name filter' }, { name: 'q', type: 'string', required: false, description: 'Search query' }, { name: 'limit', type: 'number', required: false, description: 'Max entries (default 50)' }] },
    ],
  },
  {
    category: 'Config & Agents',
    items: [
      { method: 'GET', path: '/api/modes', description: 'List agent modes' },
      { method: 'GET', path: '/api/config', description: 'Get runtime configuration' },
      { method: 'POST', path: '/api/config', description: 'Update runtime configuration', params: [{ name: 'config', type: 'object', required: true, description: 'Configuration object' }] },
      { method: 'GET', path: '/api/agents/:id', description: 'Get agent details' },
    ],
  },
  {
    category: 'Realtime',
    items: [
      { method: 'GET', path: '/api/events', description: 'SSE stream of system events' },
    ],
  },
];

const methodColors: Record<string, string> = {
  GET: 'text-emerald-400 bg-emerald-500/10 border-emerald-500/20',
  POST: 'text-[#4fc3f7] bg-[#4fc3f7]/10 border-[#4fc3f7]/20',
};

export default function ApiDocs() {
  const [expanded, setExpanded] = useState<Set<string>>(new Set(['System']));
  const [copied, setCopied] = useState<string | null>(null);

  const toggle = (cat: string) => {
    setExpanded((prev) => {
      const next = new Set(prev);
      if (next.has(cat)) next.delete(cat);
      else next.add(cat);
      return next;
    });
  };

  const copy = (text: string) => {
    navigator.clipboard.writeText(text);
    setCopied(text);
    setTimeout(() => setCopied(null), 1500);
  };

  return (
    <div className="space-y-6 max-w-3xl">
      <div className="flex items-center gap-2">
        <BookOpen size={20} className="text-aurelius-accent" />
        <h2 className="text-lg font-bold text-aurelius-text">API Documentation</h2>
      </div>

      <p className="text-sm text-aurelius-muted">
        Base URL: <code className="text-aurelius-text bg-aurelius-bg border border-aurelius-border px-1.5 py-0.5 rounded text-xs">/api</code>
      </p>

      <div className="space-y-3">
        {endpoints.map((group) => {
          const isOpen = expanded.has(group.category);
          return (
            <div key={group.category} className="aurelius-card p-0 overflow-hidden">
              <button
                onClick={() => toggle(group.category)}
                className="w-full flex items-center justify-between px-4 py-3 hover:bg-aurelius-border/10 transition-colors"
              >
                <div className="flex items-center gap-2">
                  <Server size={16} className="text-aurelius-accent" />
                  <span className="text-sm font-semibold text-aurelius-text">{group.category}</span>
                  <span className="text-[10px] text-aurelius-muted bg-aurelius-bg border border-aurelius-border px-1.5 py-0.5 rounded">
                    {group.items.length}
                  </span>
                </div>
                {isOpen ? <ChevronDown size={16} className="text-aurelius-muted" /> : <ChevronRight size={16} className="text-aurelius-muted" />}
              </button>
              {isOpen && (
                <div className="divide-y divide-aurelius-border/50">
                  {group.items.map((ep) => (
                    <div key={ep.path} className="px-4 py-3 space-y-2">
                      <div className="flex items-center gap-3">
                        <span className={`text-[10px] font-bold px-1.5 py-0.5 rounded border ${methodColors[ep.method]}`}>
                          {ep.method}
                        </span>
                        <code className="text-sm text-aurelius-text font-mono">{ep.path}</code>
                        <button
                          onClick={() => copy(ep.path)}
                          className="ml-auto text-aurelius-muted hover:text-aurelius-accent transition-colors"
                        >
                          {copied === ep.path ? <CheckCircle2 size={14} className="text-emerald-400" /> : <Copy size={14} />}
                        </button>
                      </div>
                      <p className="text-xs text-aurelius-muted">{ep.description}</p>
                      {ep.params && ep.params.length > 0 && (
                        <div className="mt-2 bg-aurelius-bg border border-aurelius-border rounded-lg p-2.5">
                          <p className="text-[10px] font-bold text-aurelius-muted uppercase tracking-wider mb-1.5">Parameters</p>
                          <div className="space-y-1">
                            {ep.params.map((p) => (
                              <div key={p.name} className="flex items-center gap-2 text-xs">
                                <code className="text-aurelius-accent font-mono">{p.name}</code>
                                <span className="text-[10px] text-aurelius-muted">{p.type}</span>
                                {p.required && <span className="text-[10px] text-rose-400">required</span>}
                                <span className="text-[10px] text-aurelius-muted">— {p.description}</span>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  ))}
                </div>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
}
