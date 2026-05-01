import { useState } from 'react';
import { Settings as SettingsIcon, Bell, Shield, Palette, Globe, Database, Key, Users, Sliders, Save } from 'lucide-react';
import Toggle from '../components/ui/Toggle';
import Select from '../components/ui/Select';
import Input from '../components/ui/Input';
import { useToast } from '../components/ToastProvider';

const SETTINGS_SECTIONS = [
  { id: 'general', label: 'General', icon: SettingsIcon },
  { id: 'notifications', label: 'Notifications', icon: Bell },
  { id: 'security', label: 'Security', icon: Shield },
  { id: 'appearance', label: 'Appearance', icon: Palette },
  { id: 'models', label: 'Models', icon: Database },
  { id: 'api', label: 'API Keys', icon: Key },
];

export default function SettingsPage() {
  const { toast } = useToast();
  const [activeSection, setActiveSection] = useState('general');
  const [settings, setSettings] = useState({
    theme: 'dark', language: 'en', fontSize: 'medium',
    notifications: true, emailAlerts: false, soundEnabled: true,
    autoUpdate: true, telemetry: false, maxConcurrency: '4',
    defaultModel: 'aurelius-1.3b', temperature: '0.7',
  });

  const update = (key: string, value: any) => setSettings(prev => ({ ...prev, [key]: value }));
  const save = () => { toast('Settings saved', 'success'); };

  return (
    <div className="flex gap-6">
      <div className="w-48 space-y-1 shrink-0">
        {SETTINGS_SECTIONS.map(s => (
          <button key={s.id} onClick={() => setActiveSection(s.id)}
            className={`w-full flex items-center gap-2 px-3 py-2 text-sm rounded-lg transition-colors ${
              activeSection === s.id ? 'bg-[#4fc3f7]/10 text-[#4fc3f7]' : 'text-[#9e9eb0] hover:text-[#e0e0e0]'
            }`}>
            <s.icon size={14} /> {s.label}
          </button>
        ))}
      </div>

      <div className="flex-1 space-y-4">
        <h2 className="text-lg font-bold text-[#e0e0e0]">{SETTINGS_SECTIONS.find(s => s.id === activeSection)?.label}</h2>

        {activeSection === 'general' && (
          <div className="aurelius-card p-6 space-y-4">
            <div className="flex items-center justify-between"><span className="text-sm text-[#e0e0e0]">Language</span><Select value={settings.language} onChange={e => update('language', e.target.value)}><option value="en">English</option><option value="zh">Chinese</option></Select></div>
            <div className="flex items-center justify-between"><span className="text-sm text-[#e0e0e0]">Font Size</span><Select value={settings.fontSize} onChange={e => update('fontSize', e.target.value)}><option value="small">Small</option><option value="medium">Medium</option><option value="large">Large</option></Select></div>
            <div className="flex items-center justify-between"><span className="text-sm text-[#e0e0e0]">Default Model</span><Select value={settings.defaultModel} onChange={e => update('defaultModel', e.target.value)}><option value="aurelius-1.3b">Aurelius 1.3B</option><option value="aurelius-2.7b">Aurelius 2.7B</option></Select></div>
          </div>
        )}

        {activeSection === 'notifications' && (
          <div className="aurelius-card p-6 space-y-4">
            {[{k:'notifications',l:'Push Notifications'},{k:'emailAlerts',l:'Email Alerts'},{k:'soundEnabled',l:'Sound Effects'}].map(item => (
              <div key={item.k} className="flex items-center justify-between"><span className="text-sm text-[#e0e0e0]">{item.l}</span><Toggle checked={(settings as any)[item.k]} onChange={v => update(item.k, v)} /></div>
            ))}
          </div>
        )}

        {activeSection === 'security' && (
          <div className="aurelius-card p-6 space-y-4">
            {[{k:'autoUpdate',l:'Automatic Updates'},{k:'telemetry',l:'Usage Telemetry'}].map(item => (
              <div key={item.k} className="flex items-center justify-between"><span className="text-sm text-[#e0e0e0]">{item.l}</span><Toggle checked={(settings as any)[item.k]} onChange={v => update(item.k, v)} /></div>
            ))}
          </div>
        )}

        <button onClick={save} className="aurelius-btn-primary flex items-center gap-2"><Save size={14} /> Save Settings</button>
      </div>
    </div>
  );
}
