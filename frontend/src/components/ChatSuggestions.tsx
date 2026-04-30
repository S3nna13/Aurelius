import { useState, useEffect, useRef } from 'react';
import { Command, Bot, Activity, Brain, Wrench, Bell, Settings } from 'lucide-react';

interface Suggestion {
  command: string;
  description: string;
  icon: typeof Command;
}

const suggestions: Suggestion[] = [
  { command: '/status', description: 'Show system status', icon: Activity },
  { command: '/agents', description: 'List active agents', icon: Bot },
  { command: '/memory', description: 'Search memory', icon: Brain },
  { command: '/skills', description: 'List available skills', icon: Wrench },
  { command: '/notifications', description: 'Show recent notifications', icon: Bell },
  { command: '/help', description: 'Show available commands', icon: Command },
  { command: '/settings', description: 'Open settings', icon: Settings },
  { command: '/clear', description: 'Clear chat history', icon: Command },
];

interface ChatSuggestionsProps {
  input: string;
  onSelect: (text: string) => void;
  visible: boolean;
}

export default function ChatSuggestions({ input, onSelect, visible }: ChatSuggestionsProps) {
  const [selectedIndex, setSelectedIndex] = useState(0);
  const containerRef = useRef<HTMLDivElement>(null);

  const filtered = input.startsWith('/')
    ? suggestions.filter((s) => s.command.startsWith(input.toLowerCase()))
    : suggestions.filter((s) => s.command.includes(input.toLowerCase()) || s.description.toLowerCase().includes(input.toLowerCase()));

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    setSelectedIndex(0);
  }, [input]);

  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (!visible || filtered.length === 0) return;
      if (e.key === 'ArrowDown') {
        e.preventDefault();
        setSelectedIndex((i) => (i + 1) % filtered.length);
      } else if (e.key === 'ArrowUp') {
        e.preventDefault();
        setSelectedIndex((i) => (i - 1 + filtered.length) % filtered.length);
      } else if (e.key === 'Enter' || e.key === 'Tab') {
        e.preventDefault();
        onSelect(filtered[selectedIndex].command + ' ');
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [visible, filtered, selectedIndex, onSelect]);

  if (!visible || filtered.length === 0 || input.length < 1) return null;

  return (
    <div
      ref={containerRef}
      className="absolute bottom-full left-0 right-0 mb-2 bg-aurelius-card border border-aurelius-border rounded-lg shadow-xl overflow-hidden z-10"
    >
      {filtered.map((s, i) => {
        const Icon = s.icon;
        return (
          <button
            key={s.command}
            onClick={() => onSelect(s.command + ' ')}
            className={`w-full flex items-center gap-3 px-3 py-2.5 text-left transition-colors ${
              i === selectedIndex ? 'bg-aurelius-accent/10' : 'hover:bg-aurelius-border/20'
            }`}
          >
            <Icon size={14} className="text-aurelius-muted shrink-0" />
            <code className="text-sm text-aurelius-accent font-mono shrink-0">{s.command}</code>
            <span className="text-xs text-aurelius-muted truncate">{s.description}</span>
          </button>
        );
      })}
    </div>
  );
}
