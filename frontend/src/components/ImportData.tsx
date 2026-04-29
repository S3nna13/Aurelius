import { useState } from 'react';
import { Upload, FileJson, FileSpreadsheet, X, CheckCircle, AlertCircle, Loader2 } from 'lucide-react';
import Modal from './ui/Modal';

interface ImportDataProps {
  onImport: (data: any[]) => void;
  onClose: () => void;
  acceptedFormats?: string[];
}

export default function ImportData({ onImport, onClose, acceptedFormats = ['json', 'csv'] }: ImportDataProps) {
  const [dragOver, setDragOver] = useState(false);
  const [file, setFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [preview, setPreview] = useState<any[] | null>(null);

  const handleFile = async (f: File) => {
    setFile(f); setError(''); setLoading(true);
    try {
      const text = await f.text();
      let data: any[];
      if (f.name.endsWith('.json')) {
        data = JSON.parse(text);
        if (!Array.isArray(data)) data = [data];
      } else if (f.name.endsWith('.csv')) {
        const lines = text.split('\n').filter(l => l.trim());
        const headers = lines[0].split(',').map(h => h.trim());
        data = lines.slice(1).map(line => {
          const vals = line.split(',').map(v => v.trim());
          return headers.reduce((obj, h, i) => ({ ...obj, [h]: vals[i] }), {} as any);
        });
      } else { throw new Error('Unsupported format'); }
      setPreview(data.slice(0, 5));
      onImport(data);
    } catch (e: any) { setError(e.message); }
    setLoading(false);
  };

  return (
    <Modal onClose={onClose}>
      <div className="space-y-4">
        <h3 className="text-lg font-bold text-[#e0e0e0] flex items-center gap-2"><Upload size={18} className="text-[#4fc3f7]" /> Import Data</h3>

        <div
          onDragOver={e => { e.preventDefault(); setDragOver(true); }}
          onDragLeave={() => setDragOver(false)}
          onDrop={e => { e.preventDefault(); setDragOver(false); const f = e.dataTransfer.files[0]; if (f) handleFile(f); }}
          className={`border-2 border-dashed rounded-xl p-8 text-center transition-colors cursor-pointer ${dragOver ? 'border-[#4fc3f7] bg-[#4fc3f7]/5' : 'border-[#2d2d44] hover:border-[#4fc3f7]/50'}`}
          onClick={() => document.getElementById('file-input')?.click()}
        >
          <Upload size={32} className="mx-auto text-[#9e9eb0] mb-3" />
          <p className="text-sm text-[#e0e0e0]">Drop a file here or click to browse</p>
          <p className="text-xs text-[#9e9eb0] mt-1">JSON or CSV</p>
          <input id="file-input" type="file" accept=".json,.csv" onChange={e => e.target.files?.[0] && handleFile(e.target.files[0])} className="hidden" />
        </div>

        {loading && <div className="flex items-center gap-2 text-sm text-[#4fc3f7]"><Loader2 size={14} className="animate-spin" /> Processing...</div>}
        {error && <div className="flex items-center gap-2 text-sm text-rose-400"><AlertCircle size={14} /> {error}</div>}
        {preview && <div className="flex items-center gap-2 text-sm text-emerald-400"><CheckCircle size={14} /> {preview.length} records ready</div>}
      </div>
    </Modal>
  );
}
