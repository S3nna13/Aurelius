import { useState, useRef } from 'react';
import { Upload, FileJson, AlertTriangle, CheckCircle, X } from 'lucide-react';
import { useToast } from './ToastProvider';

interface ImportDataProps {
  onClose: () => void;
  onImport?: (data: unknown) => void;
  title?: string;
  acceptedTypes?: string;
}

export default function ImportData({
  onClose,
  onImport,
  title = 'Import Data',
  acceptedTypes = '.json',
}: ImportDataProps) {
  const [dragOver, setDragOver] = useState(false);
  const [content, setContent] = useState('');
  const [error, setError] = useState<string | null>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const { toast } = useToast();

  const handleFile = (file: File) => {
    const reader = new FileReader();
    reader.onload = (e) => {
      const text = String(e.target?.result || '');
      setContent(text);
      setError(null);
    };
    reader.onerror = () => setError('Failed to read file');
    reader.readAsText(file);
  };

  const validateAndImport = () => {
    if (!content.trim()) {
      setError('Please paste or upload data');
      return;
    }
    try {
      const parsed = JSON.parse(content);
      if (onImport) {
        onImport(parsed);
      }
      toast('Data imported successfully', 'success');
      onClose();
    } catch {
      setError('Invalid JSON format');
    }
  };

  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/60 backdrop-blur-sm p-4">
      <div className="bg-aurelius-card border border-aurelius-border rounded-xl w-full max-w-lg shadow-2xl">
        <div className="flex items-center justify-between px-5 py-4 border-b border-aurelius-border">
          <div className="flex items-center gap-2">
            <Upload size={18} className="text-aurelius-accent" />
            <h3 className="text-base font-bold text-aurelius-text">{title}</h3>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg text-aurelius-muted hover:text-aurelius-text transition-colors"
          >
            <X size={16} />
          </button>
        </div>

        <div className="p-5 space-y-4">
          {/* Drop zone */}
          <div
            onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
            onDragLeave={() => setDragOver(false)}
            onDrop={(e) => {
              e.preventDefault();
              setDragOver(false);
              const file = e.dataTransfer.files[0];
              if (file) handleFile(file);
            }}
            className={`border-2 border-dashed rounded-lg p-6 text-center transition-colors cursor-pointer ${
              dragOver
                ? 'border-aurelius-accent bg-aurelius-accent/5'
                : 'border-aurelius-border bg-aurelius-bg hover:border-aurelius-accent/30'
            }`}
          >
            <FileJson size={24} className="mx-auto mb-2 text-aurelius-muted" />
            <p className="text-sm text-aurelius-text">Drop a file here or click to upload</p>
            <input
              type="file"
              accept={acceptedTypes}
              onChange={(e) => {
                const file = e.target.files?.[0];
                if (file) handleFile(file);
              }}
              className="hidden"
              id="import-file-input"
            />
            <label
              htmlFor="import-file-input"
              className="mt-2 inline-block text-xs text-aurelius-accent hover:underline cursor-pointer"
            >
              Browse files
            </label>
          </div>

          {/* Textarea */}
          <div>
            <label className="block text-xs text-aurelius-muted uppercase tracking-wider mb-1.5">
              Or paste JSON
            </label>
            <textarea
              ref={textareaRef}
              value={content}
              onChange={(e) => { setContent(e.target.value); setError(null); }}
              placeholder='{"key": "value"}'
              rows={6}
              className="w-full bg-aurelius-bg border border-aurelius-border rounded-lg px-3 py-2 text-sm text-aurelius-text font-mono placeholder:text-aurelius-muted focus:outline-none focus:border-aurelius-accent resize-none"
            />
          </div>

          {error && (
            <div className="flex items-center gap-2 text-xs text-rose-400 bg-rose-500/5 border border-rose-500/20 rounded-lg px-3 py-2">
              <AlertTriangle size={14} />
              {error}
            </div>
          )}

          <div className="flex justify-end gap-2">
            <button onClick={onClose} className="aurelius-btn-outline text-xs">
              Cancel
            </button>
            <button onClick={validateAndImport} className="aurelius-btn flex items-center gap-1.5 text-xs">
              <CheckCircle size={14} />
              Import
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
