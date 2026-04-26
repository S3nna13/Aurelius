import { useState, useEffect } from 'react';
import { X, ChevronRight, ChevronLeft, Sparkles } from 'lucide-react';

interface TourStep {
  target: string;
  title: string;
  description: string;
  position?: 'top' | 'bottom' | 'left' | 'right';
}

const steps: TourStep[] = [
  {
    target: 'body',
    title: 'Welcome to Aurelius',
    description: 'Your autonomous AI agent mission control. Let\'s take a quick tour.',
    position: 'bottom',
  },
  {
    target: 'nav',
    title: 'Navigation',
    description: 'Access all modules from the sidebar. Watch for live badges on notifications and workflows.',
    position: 'right',
  },
  {
    target: 'header',
    title: 'Mission Control Header',
    description: 'Toggle themes, check connection status, and access your notifications.',
    position: 'bottom',
  },
  {
    target: 'main',
    title: 'Dashboard',
    description: 'Monitor agent health, activity trends, and system metrics in real-time.',
    position: 'top',
  },
  {
    target: 'body',
    title: 'Keyboard Shortcuts',
    description: 'Press ? anytime to see shortcuts. Use Cmd+K for the command palette and / for global search.',
    position: 'bottom',
  },
];

const STORAGE_KEY = 'aurelius-tour-completed';

export default function OnboardingTour() {
  const [show, setShow] = useState(false);
  const [step, setStep] = useState(0);

  useEffect(() => {
    const completed = localStorage.getItem(STORAGE_KEY);
    if (!completed) {
      const timer = setTimeout(() => setShow(true), 800);
      return () => clearTimeout(timer);
    }
  }, []);

  const complete = () => {
    localStorage.setItem(STORAGE_KEY, 'true');
    setShow(false);
  };

  const next = () => {
    if (step < steps.length - 1) setStep((s) => s + 1);
    else complete();
  };

  const prev = () => setStep((s) => Math.max(0, s - 1));

  if (!show) return null;

  const current = steps[step];

  return (
    <div className="fixed inset-0 z-[80] flex items-center justify-center">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/50 backdrop-blur-sm" onClick={complete} />

      {/* Tooltip Card */}
      <div className="relative bg-aurelius-card border border-aurelius-border rounded-xl p-6 max-w-sm w-full shadow-2xl mx-4">
        <button
          onClick={complete}
          className="absolute top-3 right-3 p-1 rounded-lg text-aurelius-muted hover:text-aurelius-text transition-colors"
        >
          <X size={16} />
        </button>

        <div className="flex items-center gap-2 mb-3">
          <Sparkles size={18} className="text-aurelius-accent" />
          <h3 className="text-base font-bold text-aurelius-text">{current.title}</h3>
        </div>
        <p className="text-sm text-aurelius-muted leading-relaxed mb-5">
          {current.description}
        </p>

        {/* Progress */}
        <div className="flex items-center gap-1.5 mb-5">
          {steps.map((_, i) => (
            <div
              key={i}
              className={`h-1.5 rounded-full transition-all ${
                i === step ? 'w-6 bg-aurelius-accent' : 'w-1.5 bg-aurelius-border'
              }`}
            />
          ))}
        </div>

        <div className="flex items-center justify-between">
          <button
            onClick={prev}
            disabled={step === 0}
            className="flex items-center gap-1 text-xs text-aurelius-muted hover:text-aurelius-text disabled:opacity-30 transition-colors"
          >
            <ChevronLeft size={14} />
            Back
          </button>
          <button
            onClick={next}
            className="aurelius-btn flex items-center gap-1.5 text-xs"
          >
            {step === steps.length - 1 ? 'Get Started' : 'Next'}
            {step < steps.length - 1 && <ChevronRight size={14} />}
          </button>
        </div>
      </div>
    </div>
  );
}
