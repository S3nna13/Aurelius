import { useEffect, useState } from 'react';
import { Settings as SettingsIcon, Bell, Shield, Palette, Globe, Database, Key, Users, Sliders, Save } from 'lucide-react';
import Toggle from '../components/ui/Toggle';
import Select from '../components/ui/Select';
import Input from '../components/ui/Input';
import { useToast } from '../components/ToastProvider';
import { api } from '../api/AureliusClient';

const SETTINGS_SECTIONS = [
  { id: 'general', label: 'General', icon: SettingsIcon },
  { id: 'notifications', label: 'Notifications', icon: Bell },
  { id: 'security', label: 'Security', icon: Shield },
  { id: 'appearance', label: 'Appearance', icon: Palette },
  { id: 'models', label: 'Models', icon: Database },
  { id: 'api', label: 'API Keys', icon: Key },
];

const MODEL_OPTIONS = [
  { value: 'aurelius-1.3b', label: 'Aurelius 1.3B' },
  { value: 'aurelius-2.7b', label: 'Aurelius 2.7B' },
  { value: 'aurelius-3b', label: 'Aurelius 3.0B' },
  { value: 'aurelius-moe', label: 'Aurelius MoE 5B' },
];

const VALID_BACKENDS = new Set(['mock', 'vllm', 'agentic']);

export default function SettingsPage() {
  const { toast } = useToast();
  const [activeSection, setActiveSection] = useState('general');
  const [saving, setSaving] = useState(false);
  const [settings, setSettings] = useState({
    theme: 'dark', language: 'en', fontSize: 'medium',
    notifications: true, emailAlerts: false, soundEnabled: true,
    autoUpdate: true, telemetry: false, maxConcurrency: '4',
    defaultModel: 'aurelius-1.3b', temperature: '0.7',
    defaultBackend: 'mock',
    upstreamUrl: 'http://127.0.0.1:8080',
    vllmUpstreamUrl: 'http://127.0.0.1:8080',
    agenticUpstreamUrl: 'http://127.0.0.1:8080',
  });

  const update = (key: string, value: any) => setSettings(prev => ({ ...prev, [key]: value }));

  useEffect(() => {
    let active = true;
    api.getConfig()
      .then(({ config }) => {
        if (!active) return;
        setSettings(prev => ({
          ...prev,
          defaultBackend:
            typeof config['chat.backend'] === 'string' && VALID_BACKENDS.has(config['chat.backend'].trim().toLowerCase())
              ? config['chat.backend'].trim().toLowerCase()
              : prev.defaultBackend,
          defaultModel:
            typeof config['chat.model'] === 'string' && MODEL_OPTIONS.some(option => option.value === config['chat.model'].trim())
              ? config['chat.model'].trim()
              : prev.defaultModel,
          temperature: config['chat.temperature'] || prev.temperature,
          upstreamUrl: config['chat.upstream_url'] || prev.upstreamUrl,
          vllmUpstreamUrl: config['chat.vllm_upstream_url'] || prev.vllmUpstreamUrl,
          agenticUpstreamUrl: config['chat.agentic_upstream_url'] || prev.agenticUpstreamUrl,
        }));
      })
      .catch(() => {
        // Keep local defaults when config is unavailable.
      });
    return () => {
      active = false;
    };
  }, []);

  const save = async () => {
    setSaving(true);
    try {
      await api.setConfig({
        'chat.backend': settings.defaultBackend,
        'chat.model': settings.defaultModel,
        'chat.temperature': settings.temperature,
        'chat.upstream_url': settings.upstreamUrl,
        'chat.vllm_upstream_url': settings.vllmUpstreamUrl,
        'chat.agentic_upstream_url': settings.agenticUpstreamUrl,
      });
      toast('Settings saved', 'success');
    } catch (error) {
      const message = error instanceof Error ? error.message : 'Unable to save settings';
      toast(message, 'error');
    } finally {
      setSaving(false);
    }
  };

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
          </div>
        )}

        {activeSection === 'models' && (
          <div className="aurelius-card p-6 space-y-4">
            <div className="flex items-center justify-between">
              <span className="text-sm text-[#e0e0e0]">Default Model</span>
              <Select value={settings.defaultModel} onChange={e => update('defaultModel', e.target.value)}>
                {MODEL_OPTIONS.map(option => (
                  <option key={option.value} value={option.value}>
                    {option.label}
                  </option>
                ))}
              </Select>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-[#e0e0e0]">Default Backend</span>
              <Select value={settings.defaultBackend} onChange={e => update('defaultBackend', e.target.value)}>
                <option value="auto">Auto (uses first available)</option>
                <option value="mock">Mock</option>
                <option value="vllm">vLLM</option>
                <option value="agentic">Agentic</option>
              </Select>
            </div>
            <div>
              <label className="text-sm text-[#e0e0e0] block mb-1">Default Upstream URL</label>
              <Input value={settings.upstreamUrl} onChange={e => update('upstreamUrl', e.target.value)} />
            </div>
            <div>
              <label className="text-sm text-[#e0e0e0] block mb-1">vLLM Upstream URL</label>
              <Input value={settings.vllmUpstreamUrl} onChange={e => update('vllmUpstreamUrl', e.target.value)} />
            </div>
            <div>
              <label className="text-sm text-[#e0e0e0] block mb-1">Agentic Upstream URL</label>
              <Input value={settings.agenticUpstreamUrl} onChange={e => update('agenticUpstreamUrl', e.target.value)} />
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-[#e0e0e0]">Default Temperature</span>
              <Input value={settings.temperature} onChange={e => update('temperature', e.target.value)} />
            </div>
            <p className="text-xs text-[#9e9eb0]">
              Mission Control uses the Default Backend when the chat or Playground mode is set to Auto. Explicit backend selections (Mock, vLLM, Agentic) override this default.
            </p>
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

        <button onClick={save} disabled={saving} className="aurelius-btn-primary flex items-center gap-2 disabled:opacity-50">
          <Save size={14} /> {saving ? 'Saving...' : 'Save Settings'}
        </button>
      </div>
    </div>
  );
}
