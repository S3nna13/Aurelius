import { useState } from 'react';
import { Check, Copy, Download, FileJson, FileText } from 'lucide-react';

interface DataExportProps {
  data: any;
  filename?: string;
  formats?: ('json' | 'csv')[];
}

export default function DataExport({ data, filename = 'export', formats = ['json', 'csv'] }: DataExportProps) {
  const [copied, setCopied] = useState(false);

  const toCSV = (obj: any): string => {
    if (!Array.isArray(obj)) obj = [obj];
    if (obj.length === 0) return '';
    const headers = Object.keys(obj[0]);
    const rows = obj.map((row: any) => headers.map(h => {
      const val = row[h];
      if (val === null || val === undefined) return '';
      const str = String(val);
      return str.includes(',') || str.includes('"') ? `"${str.replace(/"/g, '""')}"` : str;
    }).join(','));
    return [headers.join(','), ...rows].join('\n');
  };

  const download = (format: 'json' | 'csv') => {
    const content = format === 'json' ? JSON.stringify(data, null, 2) : toCSV(data);
    const blob = new Blob([content], { type: format === 'json' ? 'application/json' : 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${filename}.${format}`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const copyJSON = async () => {
    await navigator.clipboard.writeText(JSON.stringify(data, null, 2));
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="flex items-center gap-1.5">
      {formats.includes('json') && (
        <button onClick={() => download('json')} className="aurelius-btn-outline p-1.5" title="Download JSON">
          <FileJson size={12} />
        </button>
      )}
      {formats.includes('csv') && (
        <button onClick={() => download('csv')} className="aurelius-btn-outline p-1.5" title="Download CSV">
          <FileText size={12} />
        </button>
      )}
      <button onClick={copyJSON} className="aurelius-btn-outline p-1.5" title="Copy to clipboard">
        {copied ? <Check size={12} className="text-emerald-400" /> : <Copy size={12} />}
      </button>
    </div>
  );
}
