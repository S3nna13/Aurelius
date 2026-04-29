import { useState, useEffect } from 'react';
import { HelpCircle, X, ChevronRight, ChevronLeft, Cpu } from 'lucide-react';

const STEPS = [
  { title: 'Welcome to Aurelius', content: 'Your AI agent operations center. Monitor, spawn, and manage AI agents from one dashboard.' },
  { title: 'Agent Operations', content: 'Visit the Agents page to spawn new agents, monitor their status, and view real-time execution streams.' },
  { title: 'Training & Analytics', content: 'Track model training in real-time with live loss curves, and monitor system performance in the Analytics dashboard.' },
  { title: 'Keyboard Shortcuts', content: 'Press ? to view all keyboard shortcuts. Use ⌘K for the command palette, G then D to go to Dashboard.' },
];

export default function OnboardingTour() {
  const [active, setActive] = useState(false);
  const [step, setStep] = useState(0);
  const dismissed = localStorage.getItem('onboarding_dismissed');

  useEffect(() => {
    if (!dismissed) {
      const timer = setTimeout(() => setActive(true), 1000);
      return () => clearTimeout(timer);
    }
  }, [dismissed]);

  const dismiss = () => {
    setActive(false);
    localStorage.setItem('onboarding_dismissed', 'true');
  };

  if (!active) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
      <div className="aurelius-card p-6 max-w-md w-full mx-4 space-y-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-2">
            <Cpu size={20} className="text-[#4fc3f7]" />
            <span className="text-sm font-bold text-[#e0e0e0]">{STEPS[step].title}</span>
          </div>
          <button onClick={dismiss} className="text-[#9e9eb0] hover:text-[#e0e0e0]"><X size={16} /></button>
        </div>
        <p className="text-sm text-[#9e9eb0] leading-relaxed">{STEPS[step].content}</p>
        <div className="flex items-center justify-between">
          <div className="flex gap-1.5">
            {STEPS.map((_, i) => (
              <div key={i} className={`w-2 h-2 rounded-full transition-colors ${i === step ? 'bg-[#4fc3f7]' : 'bg-[#2d2d44]'}`} />
            ))}
          </div>
          <div className="flex gap-2">
            {step > 0 && <button onClick={() => setStep(s => s - 1)} className="aurelius-btn-outline p-1.5"><ChevronLeft size={14} /></button>}
            {step < STEPS.length - 1 ? (
              <button onClick={() => setStep(s => s + 1)} className="aurelius-btn-primary p-1.5"><ChevronRight size={14} /></button>
            ) : (
              <button onClick={dismiss} className="aurelius-btn-primary text-xs px-3 py-1.5">Get Started</button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
