import { useState } from 'react';
import { motion } from 'framer-motion';
import {
  BookOpen, Search, Upload, Trash2, Plus, FileText,
  Link as LinkIcon, Globe, Database, CheckCircle, Clock, X,
} from 'lucide-react';
import Input from '../components/ui/Input';
import { useApi } from '../hooks/useApi';
import EmptyState from '../components/EmptyState';
import Skeleton from '../components/Skeleton';

interface Document {
  id: string; name: string; type: string; size: string;
  chunks: number; status: string; uploadedAt: string;
}

export default function AgentKnowledgeBase() {
  const [search, setSearch] = useState('');
  const [showAdd, setShowAdd] = useState(false);
  const [url, setUrl] = useState('');
  const { data, loading } = useApi<{ documents: Document[] }>('/knowledge', { refreshInterval: 10000 });
  const docs = (data?.documents || []).filter(d => !search || d.name.toLowerCase().includes(search.toLowerCase()));

  const addUrl = async () => {
    if (!url.trim()) return;
    await fetch('/api/knowledge', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url: url.trim(), type: 'web' }),
    });
    setUrl('');
    setShowAdd(false);
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2">
          <BookOpen size={20} className="text-[#4fc3f7]" /> Knowledge Base
          <span className="text-sm font-normal text-[#9e9eb0]">({docs.length} documents)</span>
        </h2>
        <div className="flex gap-2">
          <div className="relative">
            <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#9e9eb0]" />
            <Input value={search} onChange={e => setSearch(e.target.value)} placeholder="Search documents..." className="pl-8 py-1.5 text-sm w-48" />
          </div>
          <button onClick={() => setShowAdd(true)} className="aurelius-btn-primary flex items-center gap-2 text-sm">
            <Plus size={14} /> Add Document
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-3 gap-3">
        {[
          { label: 'Total Documents', value: docs.length, icon: FileText, color: 'text-[#4fc3f7]' },
          { label: 'Total Chunks', value: docs.reduce((sum, d) => sum + (d.chunks || 0), 0), icon: Database, color: 'text-emerald-400' },
          { label: 'Indexed', value: docs.filter(d => d.status === 'indexed').length, icon: CheckCircle, color: 'text-amber-400' },
        ].map(s => (
          <div key={s.label} className="aurelius-card text-center py-3">
            <s.icon size={16} className={`mx-auto mb-1 ${s.color}`} />
            <p className={`text-lg font-bold ${s.color}`}>{s.value}</p>
            <p className="text-[10px] text-[#9e9eb0] uppercase tracking-wider">{s.label}</p>
          </div>
        ))}
      </div>

      {loading && <Skeleton className="h-32" />}
      {!loading && docs.length === 0 && <EmptyState icon={BookOpen} title="No Documents" description="Add documents to give agents knowledge to draw from." />}

      <div className="space-y-2">
        {docs.map(doc => (
          <motion.div key={doc.id} initial={{ opacity: 0 }} animate={{ opacity: 1 }} className="aurelius-card p-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-9 h-9 rounded-lg bg-[#4fc3f7]/10 flex items-center justify-center">
                {doc.type === 'web' ? <Globe size={16} className="text-[#4fc3f7]" /> : <FileText size={16} className="text-[#4fc3f7]" />}
              </div>
              <div>
                <p className="text-sm font-medium text-[#e0e0e0]">{doc.name}</p>
                <div className="flex items-center gap-2 text-[10px] text-[#9e9eb0] mt-0.5">
                  <span>{doc.type}</span>
                  <span>·</span>
                  <span>{doc.size}</span>
                  <span>·</span>
                  <span>{doc.chunks} chunks</span>
                </div>
              </div>
            </div>
            <div className="flex items-center gap-2">
              <span className={`text-[10px] px-2 py-0.5 rounded-full ${
                doc.status === 'indexed' ? 'text-emerald-400 bg-emerald-500/10' :
                doc.status === 'processing' ? 'text-amber-400 bg-amber-500/10' :
                'text-rose-400 bg-rose-500/10'
              }`}>{doc.status}</span>
            </div>
          </motion.div>
        ))}
      </div>

      {showAdd && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm p-4" onClick={() => setShowAdd(false)}>
          <div className="aurelius-card p-6 max-w-md w-full space-y-4" onClick={e => e.stopPropagation()}>
            <h3 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2"><Plus size={18} className="text-[#4fc3f7]" /> Add Document</h3>
            <div className="border-2 border-dashed border-[#2d2d44] rounded-xl p-8 text-center cursor-pointer hover:border-[#4fc3f7]/50 transition-colors">
              <Upload size={24} className="mx-auto text-[#9e9eb0] mb-2" />
              <p className="text-sm text-[#9e9eb0]">Drop files or click to upload</p>
              <p className="text-xs text-[#2d2d44] mt-1">PDF, TXT, MD, CSV</p>
            </div>
            <div className="flex gap-2">
              <div className="relative flex-1">
                <Globe size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-[#9e9eb0]" />
                <Input value={url} onChange={e => setUrl(e.target.value)} placeholder="Or paste a URL..." className="pl-8 py-1.5 text-sm w-full" />
              </div>
              <button onClick={addUrl} className="aurelius-btn-primary text-sm px-4">Add URL</button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
