import { Sparkles, Code, Search, Pen, Brain } from 'lucide-react';

const SUGGESTIONS = [
  { icon: Code, text: 'Write a Python function to sort a list', category: 'coding' },
  { icon: Search, text: 'Research the latest advances in LLM architecture', category: 'research' },
  { icon: Pen, text: 'Draft a technical blog post about AI agents', category: 'writing' },
  { icon: Brain, text: 'Explain how transformer attention works', category: 'learning' },
];

interface ChatSuggestionsProps {
  onSelect: (text: string) => void;
}

export default function ChatSuggestions({ onSelect }: ChatSuggestionsProps) {
  return (
    <div className="space-y-2">
      <p className="text-xs text-[#9e9eb0] flex items-center gap-1.5">
        <Sparkles size={12} className="text-[#4fc3f7]" /> Suggested prompts
      </p>
      <div className="grid grid-cols-2 gap-2">
        {SUGGESTIONS.map((s, i) => {
          const Icon = s.icon;
          return (
            <button key={i} onClick={() => onSelect(s.text)}
              className="text-left aurelius-card p-3 hover:border-[#4fc3f7]/30 transition-all group"
            >
              <Icon size={14} className="text-[#4fc3f7] mb-1.5 group-hover:scale-110 transition-transform" />
              <p className="text-xs text-[#9e9eb0] group-hover:text-[#e0e0e0] transition-colors line-clamp-2">{s.text}</p>
            </button>
          );
        })}
      </div>
    </div>
  );
}
