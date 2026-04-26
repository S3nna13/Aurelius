import { Trash2, Download } from 'lucide-react';

interface BulkActionsBarProps {
  selectedCount: number;
  totalCount: number;
  onSelectAll: () => void;
  onDeselectAll: () => void;
  onDelete: () => void;
  onExport?: () => void;
  labels?: {
    selectAll?: string;
    deselectAll?: string;
    delete?: string;
    export?: string;
  };
}

export default function BulkActionsBar({
  selectedCount,
  totalCount,
  onSelectAll,
  onDeselectAll,
  onDelete,
  onExport,
  labels = {},
}: BulkActionsBarProps) {
  if (totalCount === 0) return null;

  const allSelected = selectedCount === totalCount && totalCount > 0;

  return (
    <div className="flex items-center gap-2 px-3 py-2 bg-aurelius-surface border border-aurelius-border rounded-lg">
      <button
        onClick={allSelected ? onDeselectAll : onSelectAll}
        className="text-xs font-medium text-aurelius-muted hover:text-aurelius-text transition-colors"
      >
        {allSelected ? labels.deselectAll || 'Deselect All' : labels.selectAll || 'Select All'}
      </button>
      {selectedCount > 0 && (
        <>
          <span className="text-xs text-aurelius-muted">
            {selectedCount} selected
          </span>
          <div className="flex-1" />
          {onExport && (
            <button
              onClick={onExport}
              className="flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium bg-aurelius-surface-hover text-aurelius-text rounded hover:bg-aurelius-border transition-colors"
            >
              <Download className="w-3.5 h-3.5" />
              {labels.export || 'Export'}
            </button>
          )}
          <button
            onClick={onDelete}
            className="flex items-center gap-1.5 px-2.5 py-1 text-xs font-medium bg-red-500/10 text-red-400 rounded hover:bg-red-500/20 transition-colors"
          >
            <Trash2 className="w-3.5 h-3.5" />
            {labels.delete || 'Delete'}
          </button>
        </>
      )}
    </div>
  );
}
